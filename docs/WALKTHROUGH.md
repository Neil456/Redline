# Redline walkthrough (interview guide)

Open `demo/index.html` (or the hosted copy) and follow the sections in order.
Each section has a talk track, a live figure, and a link to the code behind it.
Plan on about 8–10 minutes. Spend longer wherever the interviewer digs in.

## The 30-second pitch

> Redline is a C++/CUDA inference engine for Llama-family models. I wrote the
> runtime myself: paged KV cache, continuous batching, GPU sampling, INT8/INT4
> weight-only kernels, CUDA graphs and speculative decoding. A benchmark
> harness compares it with Hugging Face Transformers and vLLM on the same
> workloads. Decode on one GPU is memory-bound, so most of the work goes into
> moving fewer bytes per token and keeping the GPU from ever waiting on the CPU.

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

### 06 Real run (2 min, once you have a trace)
- These numbers are measured by `redline_bench` on the GPU, with CUDA-event
  step times. Point at the prefill steps (orange) vs decode steps (blue), the
  batch filling up, and the KV pool usage tracking it.
- Hover a request in the timeline to show its TTFT and preemptions.

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
byte on the critical path. vLLM is one of the baselines, and the
benchmark harness runs both on identical workloads.

**What would you do next?** A tensor-core (mma.sync) prefill attention, a BF16
path, overlapping prefill and decode on separate streams, and async
scheduling for speculative decoding.

## Producing a real trace for section 06

```bash
build/redline_bench --model llama-int4.rlm --synthetic 32,512,256 --max-seqs 8 \
    --track-memory --trace-out trace.json
python demo/build_demo.py --trace trace.json      # bakes the trace into demo/index.html
```

You can also load any trace with the **Load trace.json** button on the page.
Keep the numbers honest: everything you quote should come from
`redline_bench`, `redline.report` or `redline.eval_quality` output on your
hardware.
