# RTX 4090 results

Measured on 2026-10-14 on a rented Runpod RTX 4090 (24 GB, 450 W, driver 580.126.20, 16 vCPU, Secure Cloud,
EU-RO-1). Redline was built with CUDA 12.8 for sm_89 with CUTLASS 3.5.1; the TensorRT-LLM backend was not
built. Baselines: Hugging Face Transformers 5.19 (PyTorch 2.8) and vLLM 0.31.0 (PyTorch 2.13). Total GPU
time was about 75 minutes. The raw per-run JSON is in this directory, and `quick/REPORT.md` and
`sweep/REPORT.md` are generated from it with `python -m redline.report`.

**Models.** Llama 3.1 8B Instruct (target) and Llama 3.2 1B Instruct (draft). The checkpoints are the
ungated re-uploads `NousResearch/Meta-Llama-3.1-8B-Instruct` and `unsloth/Llama-3.2-1B-Instruct` of Meta's
weights. vLLM INT4 uses `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` (AWQ, Marlin kernels).

| Redline config | weights | lm_head | KV cache | file size |
|---|---|---|---|---:|
| `fp16` | FP16 | FP16 | FP16 | 16.1 GB |
| `int8` | INT8 per-channel (W8A16) | INT8 | FP16 | 8.6 GB |
| `int8_kvint8` | INT8 per-channel | INT8 | INT8 per (token, head) | 8.6 GB |
| `int4` | INT4 g128 asymmetric, clip search (W4A16) | INT8 | FP16 | 5.3 GB |
| `int4_spec` | `int4` target + `int4` 1B draft, k = 4 | INT8 | FP16 | 5.3 + 1.3 GB |

Every system runs the same workload files: identical prompt token ids and output lengths, with greedy
decoding and EOS ignored. Offline workloads run as successive static batches of `batch_size`, and TTFT is
measured from batch submission. Decode throughput is `batch_size × 1000 / TPOT p50`, where TPOT is a
request's mean time between output tokens.

## Correctness on the GPU

- `redline_cuda_tests`: **24/24 pass**. This covers every kernel against a CPU reference, all GEMM paths,
  the sampling distributions, lossless speculative verification, and the full engine against the fp32 CPU
  reference model (greedy, quantized, INT8 KV, preemption, speculative). Two failures on the first run
  were fixed: cuBLASLt rejects an FP16 bias epilogue with FP32 output.
- Real checkpoints vs Transformers FP16 on the same GPU:
  - Llama 3.2 1B FP16: greedy output matches all 64 tokens on 4 of 4 chat prompts, with mean |Δ log p|
    of 0.0027.
  - Llama 3.1 8B FP16: mean |Δ log p| is 0.0024 and perplexity on a probe text is 10.820 vs 10.816. On
    one of 4 prompts greedy output diverges at token 44, on an fp16 near-tie.

## Quality retention (vs FP16 Transformers)

WikiText-2 perplexity (256k tokens) plus 250 examples each of HellaSwag, ARC-Easy and ARC-Challenge, all
scored with the same teacher-forced log-likelihoods (`redline.eval_quality`).

| model | retention | WikiText ppl (HF 8.108) | HellaSwag acc_norm (HF 0.700) | ARC-E acc (HF 0.796) | ARC-C acc_norm (HF 0.528) |
|---|---:|---:|---:|---:|---:|
| `int8` | **99.6%** | 8.108 | 0.704 | 0.796 | 0.516 |
| `int4` | 94.8% | 9.068 | 0.680 | 0.748 | 0.520 |

INT4 round-to-nearest g128 costs about 12% WikiText perplexity on Llama 3 8B. That is in line with
published RTN results for this model family, which is known to be quantization-sensitive. AWQ/GPTQ-style
calibration would recover most of it.

## Decode throughput vs batch size (512 prompt / 256 output tokens)

Each cell is TPOT p50 in ms, then decode tok/s.

| batch | Redline FP16 | Redline INT8 | Redline INT4 | Redline INT4 + spec* | HF FP16 | vLLM FP16 | vLLM AWQ INT4 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.12 / 58 | 9.27 / 108 | **5.92 / 169** | 2.90 / 344 | 19.83 / 50 | 16.72 / 60 | 6.10 / 164 |
| 2 | 17.31 / 116 | 9.34 / 214 | **6.00 / 333** | 3.76 / 531 | 20.66 / 97 | 16.98 / 118 | 6.27 / 319 |
| 4 | 17.43 / 230 | 9.49 / 422 | **6.23 / 642** | 6.01 / 666 | 21.09 / 190 | 17.14 / 233 | 6.40 / 625 |
| 8 | 18.20 / 439 | 10.56 / 758 | 9.06 / 883 | 8.77 / 913 | 21.96 / 364 | 17.42 / 459 | **6.78 / 1181** |
| 16 | 19.57 / 818 | 18.46 / 867 | 17.34 / 923 | 12.79 / 1251 | 23.85 / 671 | 19.07 / 839 | **8.38 / 1908** |
| 32 | 23.95 / 1336 | 23.32 / 1372 | 34.61 / 925 | 19.81 / 1615 | 29.28 / 1093 | 22.56 / 1419 | **11.60 / 2758** |

\* Prompts in this sweep are random token ids. The models' continuations of them are repetitive, so draft
acceptance is unrealistically high (0.71–0.84). On WikiText prompts acceptance is 0.40–0.52; see below.

Geometric mean over the six batch sizes:

| | vs HF FP16 | vs vLLM FP16 | vs vLLM AWQ INT4 |
|---|---:|---:|---:|
| Redline FP16 | +20% | −3% | −61% |
| Redline INT8 | **+81%** | +46% | −41% |
| Redline INT4 | +119% | +77% | −29% |

## Quick suite (prompts from WikiText-2)

Decode tok/s on the offline workloads:

| config | b1, 128 in | b1, 1024 in | b8, 128 in | b8, 1024 in | peak VRAM b8/1024 in (GiB) |
|---|---:|---:|---:|---:|---:|
| Redline FP16 | 60 | 58 | 462 | 400 | 16.99 |
| Redline FP16, no graphs, sync | 59 | 57 | 456 | 393 | 16.98 |
| Redline INT8 | 110 | 108 | 830 | 642 | 10.00 |
| Redline INT8 + INT8 KV | 108 | 105 | 818 | 629 | 9.38 |
| Redline INT4 | **174** | **168** | 987 | 724 | **6.96** |
| Redline INT4 + spec (accept 0.40–0.52) | 217 | 225 | 747 | 536 | 8.65 |
| HF FP16 | 51 | 50 | 380 | 344 | 18.37 |
| vLLM FP16 | 60 | 60 | 469 | 412 | (pool) |
| vLLM AWQ INT4 | 164 | 162 | **1276** | **925** | (pool) |

Geometric mean over the 4 offline workloads (`quick/REPORT.md`):

| | decode vs HF FP16 | decode vs vLLM FP16 | peak VRAM vs HF FP16 |
|---|---:|---:|---:|
| Redline FP16 | +18% | −2% | −1% |
| Redline INT8 | +109% | +74% | **−43%** |
| Redline INT8 + INT8 KV | +105% | +71% | **−45%** |
| Redline INT4 | +181% | +134% | −62% |

vLLM preallocates `gpu_memory_utilization × VRAM`, so its peak VRAM is not compared.

**Online, Poisson arrivals at 4 req/s, 50 requests, up to 32 concurrent:**

| system | TTFT p50 / p99 (ms) | TPOT p50 (ms) | output tok/s |
|---|---:|---:|---:|
| Redline FP16 | 69.8 / 195.6 | 21.91 | 603 |
| Redline INT8 | 68.5 / 200.5 | 20.30 | 666 |
| Redline INT4 | 85.9 / 208.0 | 23.04 | 670 |
| Redline INT4 + spec (accept 0.66) | 72.6 / 232.9 | 15.82 | 752 |
| HF FP16 (static batches, no continuous batching) | 1507.9 / 1888.5 | 41.59 | 139 |
| vLLM FP16 | 74.9 / 175.2 | 20.61 | 610 |
| vLLM AWQ INT4 | 42.4 / 150.4 | 7.36 | 746 |

## What the numbers say

1. **Decode is bandwidth-bound and Redline's GEMV path is close to the limit.** At batch 1 an FP16 step
   reads 15.0 GB of weights; at 1008 GB/s that is a 14.9 ms floor, and Redline takes 16.8 ms (89% of peak
   bandwidth). For INT8 the figures are 7.5 GB, 7.45 ms floor and 9.1 ms (82%); for INT4, 4.2 GB, 4.2 ms
   floor and 5.8 ms (73%). The fixed per-step costs (attention, norms, sampling) weigh more as weights
   shrink. Redline's INT4 GEMV is 3–4% faster than vLLM's Marlin kernel at batch 1–4.
2. **FP16 Redline is on par with vLLM**: 3% slower on average (2% at small batch, 6% at batch 32). It is
   16–22% faster than Transformers.
3. **Weight-only INT4 on CUDA cores stops scaling at batch 8.** From batch 16 up, the W4A16 GEMV is
   limited by FMA and L1 throughput, not DRAM. Marlin's tensor-core W4A16 GEMM keeps scaling: 11.6 ms vs
   34.6 ms at batch 32. Reordering the GEMV to unpack each weight once (in this run) helped only slightly
   (about 4% at batch 8).
   The real fix is a tensor-core mixed-input GEMM, which the TensorRT-LLM backend provides but which was
   not built on this pod.
4. **Prefill is the weak spot.** At batch 32 × 512 tokens, vLLM's TTFT is 227 ms against Redline's
   1004 ms. Redline's prefill attention is a tiled CUDA-core kernel, and a tensor-core
   (FlashAttention-style) kernel is the next step.
5. **CUDA graphs and async scheduling matter little for an 8B model** (16.80 vs 17.00 ms per step): the
   GPU work per step is far larger than the launch overhead. They matter for the small draft model, which
   runs k rounds per step.
6. **Speculative decoding helps at low batch and hurts at batch 8.** On WikiText prompts the 1B draft is
   accepted 40–52% of the time, giving 1.25–1.34× at batch 1, and online TPOT drops from 23.0 to 15.8 ms
   (66% acceptance). At batch 8,
   verifying k + 1 = 5 positions for 8 sequences (40 rows) outgrows the GEMV path, and it ends up slower
   than plain INT4.
7. **INT8 KV** halves KV memory (peak −45% vs −43% for INT8 weights alone) at a 1–2% decode cost at these
   context lengths.

## Reproduce

```bash
scripts/run_rtx4090.sh                                   # full pipeline (needs an RTX 4090 class GPU)
python -m redline.report --results results/rtx4090/quick --baseline hf_fp16 --baseline vllm \
    --baseline vllm_awq_int4 --quality results/rtx4090/quality.json
```
