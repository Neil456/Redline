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
> - INT4 beats vLLM's Marlin kernel by 5–6% at batch 1–4. I wrote a
>   tensor-core INT4 kernel that took batch 32 from 35% to 80% of Marlin.
> - AWQ + GPTQ calibration cuts INT4's perplexity loss from 11.8% to 4.4%.

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
  - Batch 1–4: Redline INT4 is the fastest system, 5–6% ahead of vLLM AWQ.
    That is the CUDA-core GEMV, with SwiGLU fused into the gate/up projection.
  - The dotted grey line is where INT4 used to stop: the GEMV flattens at
    about 950 tok/s after batch 8. Explain why: it multiplies every weight
    by every row on CUDA cores, so past ~8 rows it is FMA- and L1-bound, not
    memory-bound.
  - The solid red line is the fix: a W4A16 GEMM on tensor cores
    (`mma.sync`), 2,206 tok/s at batch 32 (80% of Marlin, up from 35%). In
    steady-state decode steps it matches Marlin at batch 8.
  - Name the remaining gap yourself. The kernel loads activation fragments
    per warp from L1 instead of staging them in shared memory the way Marlin
    does. Redline's slower prefill also stretches the first steps of each
    batch. Knowing exactly where it stops, with a profile to show it, is the
    strongest slide.
- **Bandwidth table.** At batch 1, FP16 runs at 89% of DRAM bandwidth. INT4
  reaches 74% because the fixed per-step work (attention, norms, sampling,
  launches) weighs more once the weights shrink 3.5×.
- **Quality.** INT8 is lossless in practice (WikiText perplexity 8.108 for
  both). INT4 round-to-nearest costs 11.8% perplexity on Llama 3. The
  converter's calibration fixes most of that:
  - AWQ: activation-aware per-channel scales, folded into the RMSNorm.
  - GPTQ: rounding error pushed onto the not-yet-quantized columns through
    the inverse Hessian.
  - Together the loss drops to 4.4% (mean retention 94.8% → 96.4%), at the
    same speed and file size.
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
| Quantized GEMV | `src/cuda/kernels/gemv_weight_only.cu` | `gemv_w4a16_kernel`, `store_swiglu` (fused SwiGLU output) |
| Tensor-core INT4 GEMM | `src/cuda/kernels/gemm_w4a16_tc.cu`, `src/core/int4_layout.cpp` | `dequant_word` (lop3 + `0x6400` trick), `mma_16816`, the split-K last-CTA reduction; the load-time re-layout that makes one 16-byte load = eight B fragments |
| GEMM dispatch | `src/cuda/gemm/linear.cpp` | GEMV vs tensor-core INT4 vs CUTLASS vs cuBLASLt W8A8 vs TensorRT-LLM vs dequant; `forward_gated` picks the fused SwiGLU path |
| CUTLASS backend | `src/cuda/gemm/cutlass_gemm.cu` | mixed-input FP16×INT8 tensor-core GEMM |
| TensorRT-LLM backend | `src/cuda/gemm/trtllm_gemm.cpp` | weight re-layout + per-shape config profiling |
| Sampling | `src/cuda/kernels/sampling.cu` | `radix_select_topk`, `gumbel_sample`, `speculative_verify_kernel` |
| Benchmark harness | `src/cuda/bench/harness.cpp` | metric definitions, `write_trace` |
| Quantizer | `python/redline/quantize.py` | INT4 groups with clip search |
| AWQ / GPTQ | `python/redline/calibrate.py` | `awq_search` + scale folding, `gptq_quantize` (inverse-Hessian error feedback), the layer-by-layer loop |
| Per-op profiler | `src/cuda/op_profiler.h`, `LlamaModel::forward` | CUDA-event `mark()` calls between ops |
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
of vLLM. Redline INT4 is 5–6% faster than vLLM's Marlin at batch 1–4, level at
batch 8 in steady state, and 20% behind at batch 32. vLLM's prefill is about
5× faster at large batch.

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

**How does the tensor-core INT4 kernel work?** Point at the two code excerpts:
- **Layout.** Weights are re-laid out once at load, so each lane's 16-byte
  load is exactly its B fragments for eight 8-column tiles.
- **Dequantization.** A nibble becomes an fp16 number by OR-ing it into the
  mantissa of 1024.0 (`lop3`) and subtracting 1024. One `hfma2` then applies
  the group scale and offset.
- **Pipeline.** The next chunk's weights load while this one computes.
- **Split-K.** K is split across CTAs to fill the GPU. The last CTA to finish
  a column block (an atomic ticket) adds the partials in a fixed order. The
  split count depends only on the layer shape, so a sequence gets the same
  logits whatever batch it is in.
- **Fused SwiGLU.** Gate and up columns are interleaved, so the epilogue
  writes SiLU(gate)·up directly.

**How did you tune it without Nsight Compute?** The container had no
performance-counter access, so I added a per-op profiler (CUDA events between
ops) and ran controlled A/B runs on the pod.
- The profile showed a constant 5–8 µs extra per GEMM at batch 1. That was
  the split-K reduction: one dependent L2 load chain per output. Issuing all
  of a split's loads together fixed it.
- A sweep of the split target found one CTA per SM best at every batch size.
- The profile then showed the INT8 lm_head at 11% of the step at batch 32.
  Doubling its rows per GEMV launch halved that.

**What went wrong in the second GPU session?** Three things, all caught by
controls rather than by luck:
- **Wrong kernel at batch 1.** The first INT4 sweep was 12% slower at batch
  1. `auto` picks the weight layout once per engine, and an engine sized for
  32 sequences had chosen tensor cores. Running the GEMV-only mode on the same
  pod separated "slower machine" from "wrong kernel".
- **vLLM wouldn't start.** The pod's driver predated CUDA 13. I installed
  vLLM's CUDA 12.9 build and ran it on the same pod; it reproduced the first
  pod's numbers to 0.1 ms.
- **A metric mix-up.** For a while I compared vLLM's per-step ITL against the
  report's TPOT. Checking the raw JSON caught it before the numbers went into
  the write-up.

**What would you do next?** In order of measured impact:
1. Stage activations through shared memory in the INT4 GEMM (`cp.async` +
   `ldmatrix`, more warps per activation tile) to close the last 20% to Marlin
   at batch 32.
2. A tensor-core flash-attention prefill kernel. Prefill is 5× slower than
   vLLM's and also stretches the first decode steps of a batch.
3. Decode attention at large batch: it is now the second-largest item in the
   batch-32 profile.
4. Keep speculative verification on the tensor-core kernel past 64 rows, and
   turn speculation off adaptively at high batch, where it currently loses.

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
