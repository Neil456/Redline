# Redline walkthrough (interview guide)

Run `python3 demo/open_demo.py` from the repository root (it opens `demo/index.html`, which works offline
and needs no GPU), or open the hosted copy, and follow the sections in order.
Each section has a talk track, a live figure, and a link to the code behind it.
Plan on about 8–10 minutes. Spend longer wherever the interviewer digs in.

## The 30-second pitch

> Redline is a C++/CUDA inference engine for Llama-family models. I wrote the
> runtime myself: paged KV cache, continuous batching, GPU sampling, INT8/INT4
> weight-only kernels, CUDA graphs and speculative decoding. A benchmark
> harness compares it with Hugging Face Transformers and vLLM on the same
> workloads. Decode on one GPU is memory-bound, so most of the work goes into
> moving fewer bytes per token and keeping the GPU from ever waiting on the CPU.
> On an RTX 4090 with Llama 3.1 8B:
> - INT8 decodes 81% faster than Transformers FP16 with 43% less peak memory
>   and 99.6% quality retention.
> - FP16 is on par with vLLM.
> - At batch 1, INT4 beats vLLM's Marlin kernel.

## Talk track by section

### 01 One decode step (start here, 2 min)
Click through the stages left to right. The point to land:
**the host builds one compact batch description, uploads it with a single copy,
and the GPU runs the whole step, including sampling, without coming back to
the CPU.** Sampled ids feed the next step on the device.

- Click **Paged attention** and point at the inner loop: one CTA covers all
  query heads that share a KV head (GQA), so each K vector is loaded once per
  group, which is 4× less KV traffic for Llama 3.
- Click **QKV GEMV**: one 16-byte load brings 32 INT4 weights. The group
  scale and offset apply once per 32 values via `s·Σx·q + o·Σx`.

### 02 Async scheduling + CUDA graphs (1–2 min)
- Default (8B target): toggle **Sync** and **Async**. The red hatched gaps are
  GPU idle time while the host reads tokens and plans the next step.
- Switch the model to **1B draft**, set **Eager**, then try each combination. With
  async alone the step becomes CPU-launch-bound. With graphs alone the GPU
  still waits on the host. You only get the speedup with both (1.43× in the
  model's table).
- Be explicit that this is a model with stated inputs. The measured numbers
  are in section 06.

### 03 Paged KV cache (2 min)
- Press **Play**. Point out that block tables are non-contiguous lists of block
  ids, like `[3, 4, 5, 6, 7, 8, 24]`, and memory grows one 16-token block at a
  time.
- Drag **KV pool** down to ~20 blocks and press **Play** again to trigger
  preemption. The newest sequence is evicted and recomputed, so older ones
  keep their latency.
- Use the stat "blocks a max-length contiguous reservation would need" to
  explain why paging matters.

### 04 Quantization and bandwidth (1–2 min)
- At batch 1 the bars are almost all weights. INT4 moves about 3× fewer bytes
  than FP16, so its ceiling is about 3× higher.
- Raise batch and context: KV reads take over. Flip **Per head** to show what
  GQA grouping saves, and **INT8** KV to halve it again.
- Memory chart: FP16 8B weights alone are 16 GB of a 24 GB card. INT4 leaves
  room for a much larger KV cache.

### 05 Speculative decoding (1–2 min)
- Press **Play**. Green ✓ = accepted draft token, ✗ = rejected, blue = the
  target's own token (correction or bonus).
- The guarantee to state: acceptance is `min(1, p_target/p_draft)` and
  rejections resample from `max(0, p_target − p_draft)`. The output
  distribution is exactly the target's, and greedy output is identical.
- Use the chart to show that the best `k` depends on acceptance rate and the
  draft/target cost ratio.

### 06 Measured on an RTX 4090 (3 min)
Everything in this section was measured on one RTX 4090 (`results/rtx4090/`).
The baselines ran the same workload files.
- **Batch sweep chart.** Hover each batch size.
  - Below batch 8, solid Redline lines sit at or above the dashed baselines.
  - Redline INT4 (red) flattens after batch 8 while vLLM AWQ keeps climbing.
    Say why: the W4A16 GEMV runs on CUDA cores. Up to batch 8 it is
    memory-bound, beyond that it is FMA- and L1-bound. Marlin uses tensor cores.
    This is the most honest and interesting slide: you know exactly where the
    design stops scaling and what fixes it.
- **Bandwidth table.** At batch 1, FP16 runs at 89% of DRAM bandwidth. INT4
  reaches only 73% because the fixed per-step work (attention, norms, sampling,
  launches) weighs more once the weights shrink 3.5×.
- **Quality.** INT8 is lossless in practice (WikiText perplexity 8.108 for both).
  INT4 RTN g128 costs about 12% perplexity on Llama 3. AWQ/GPTQ calibration is
  the known fix.
- **Traces.** Switch between the three recorded runs:
  - *INT4 weights:* the batch fills to about 29 sequences and the queue never
    builds.
  - *INT8, small KV pool:* the pool hits 100%. Requests wait (orange line), and
    the Gantt chart shows the light "waiting" segments of the queued ones. That
    is admission control and recompute preemption working (7 preemptions).
  - *INT4 + speculative:* far fewer steps (520 vs 1332 for the same work),
    because each step emits several tokens per sequence. Acceptance is 65.5%.

### 07 Code map
Use it when someone asks "where is that in the code?" Every excerpt is
extracted from the repository when the page is built, so paths and line
numbers match the source.

## Files to show

| Topic | File | What to point at |
|---|---|---|
| Scheduler | `src/core/scheduler.cpp` | `schedule_decode` / `schedule_prefill` lambdas, preemption loop |
| KV blocks | `src/core/block_allocator.cpp` | LIFO free list |
| Batch metadata | `src/core/batch_builder.cpp` | `add()`: slot = block_table[pos/16]·16 + pos%16 |
| Engine loop | `src/cuda/engine/engine.cpp` | `step()` async keep-one-in-flight, `upload_batch`, `launch_step`, `capture`, `run_spec_step` |
| Transformer layer | `src/cuda/model/llama.cpp` | `forward()` layer loop |
| Attention | `src/cuda/kernels/attention.cu` | `decode_attention_kernel` (GQA, split-K), `prefill_attention_kernel` (tiled, online softmax) |
| RoPE + KV write | `src/cuda/kernels/rope_kv.cu` | INT8 KV quantization per (token, head) |
| Quantized GEMV | `src/cuda/kernels/gemv_weight_only.cu` | `gemv_w4a16_kernel` |
| GEMM dispatch | `src/cuda/gemm/linear.cpp` | GEMV vs CUTLASS vs cuBLASLt W8A8 vs TensorRT-LLM vs dequant |
| CUTLASS backend | `src/cuda/gemm/cutlass_gemm.cu` | mixed-input FP16×INT8 tensor-core GEMM |
| TensorRT-LLM backend | `src/cuda/gemm/trtllm_gemm.cpp` | weight re-layout + per-shape config profiling |
| Sampling | `src/cuda/kernels/sampling.cu` | `radix_select_topk`, `gumbel_sample`, `speculative_verify_kernel` |
| Benchmark harness | `src/cuda/bench/harness.cpp` | metric definitions, `write_trace` |
| Quantizer | `python/redline/quantize.py` | INT4 groups with clip search |
| Exporter | `python/redline/convert.py` | streaming HF → `.rlm` conversion |
| Tests | `tests/python/test_convert.py`, `tests/cuda/` | exporter vs HF logits; kernels and engine vs the CPU reference |
| Measured results | `results/rtx4090/README.md` | methodology, batch sweep, quality, what the numbers say |

## Questions you should expect

**Why is decode memory-bound, and what does that imply?** Each step multiplies
a few activation rows by every weight matrix. The arithmetic intensity is tiny,
so step time ≈ bytes moved ÷ bandwidth. Everything that cuts bytes (INT4
weights, INT8 KV, GQA-grouped KV reads, fused kernels) speeds up decode.
Prefill is compute-bound, so it goes to tensor-core GEMMs instead.

**Why weight-only quantization instead of W8A8 everywhere?** At decode the
weights dominate the traffic and activations are tiny, so quantizing
activations buys little and costs accuracy (outlier channels). W8A8 through
cuBLASLt is available as a prefill option.

**How do you know the GPU code is correct?** A fp32 CPU reference model reads
the same `.rlm` file. Python tests check that the reference reproduces Hugging
Face logits, and the GPU tests compare every kernel and the full engine
against it. For greedy decoding, the end-to-end check requires each
generated token to be within a small margin of the reference argmax. That
tolerates fp16 rounding on near-ties but catches real bugs.

**Is speculative decoding lossless?** Yes. The verification kernel implements
the standard rejection-sampling scheme. `test_sampling.cu` checks it
statistically: with a deliberately mismatched draft, the first emitted token
still follows the target distribution.

**Why not just use vLLM?** The point was control and understanding every
byte on the critical path. vLLM is one of the baselines, and the benchmark
harness runs both on identical workloads. Measured: FP16 decode is within 3%
of vLLM. Redline's INT4 GEMV is 3–4% faster than vLLM's Marlin at batch 1–4
and up to 3× slower at batch 32. vLLM's prefill is about 4× faster at large
batch.

**How did you make the comparison fair?** Every system reads the same
workload JSON: identical prompt token ids, output lengths, greedy decoding,
EOS ignored. Offline workloads run as successive static batches for everyone,
with TTFT measured from batch submission. Decode throughput is
batch × 1000 / TPOT for every system. ITL would not work: speculative decoding
emits tokens in bursts, so its median ITL is 0. Online workloads are compared
on latency only, because their throughput is set by the arrival rate. I fixed
the first two of these after the first GPU run showed Redline's TTFT counting
queue time that HF's did not. That is a good story about checking your own
benchmark.

**What broke when you first ran on a real GPU?** 22 of 24 tests passed. The two
failures were one bug: cuBLASLt has no algorithm for an FP16 bias epilogue with
FP32 output. Running real checkpoints found one more bug in the Python binding,
and the benchmarks found the INT4 scaling cliff.

**What would you do next?** In order of measured impact:
1. A tensor-core W4A16 GEMM (Marlin-style, or build the TensorRT-LLM backend)
   for INT4 at batch ≥ 8.
2. A tensor-core flash-attention prefill kernel for TTFT.
3. Calibrated INT4 (AWQ/GPTQ) to recover the 5% quality gap.
4. Turning speculative decoding off adaptively at higher batch, where it
   currently loses.

## Producing traces for section 06

The page bakes in `results/rtx4090/` and the three traces in `demo/traces/`.
To record a new one:

```bash
build/redline_bench --model llama-int4.rlm --online --workload online_rate4_greedy.json \
    --max-seqs 32 --track-memory --trace-out trace.json
python demo/build_demo.py --trace "My run=trace.json"     # replaces the baked-in traces
```

You can also load any trace with the **Load another trace** button on the page.
Keep the numbers honest: everything you quote should come from
`redline_bench`, `redline.report` or `redline.eval_quality` output on your
hardware.
