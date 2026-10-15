# RTX 4090 results

Measured on 2026-10-14 on two rented Runpod RTX 4090s (24 GB, 450 W).

- **First pod:** driver 580.126.20, 16 vCPU, EU-RO-1. Every configuration and both baselines, about 75
  minutes of GPU time.
- **Second pod:** driver 570.195.03, 48 vCPU. The INT4 work: the tensor-core kernel, fused SwiGLU,
  AWQ/GPTQ calibration and quality. It re-measured every INT4 configuration and vLLM AWQ, and took
  about 80 minutes.

The two machines measure the same. On the second pod, the previous commit's binary reproduces the first
pod's INT4 numbers within 2% (`control/`). vLLM AWQ reproduces its own to within 0.1 ms per token at
every batch size. So rows from the two pods can be compared.

Redline was built with CUDA 12.8 for sm_89 with CUTLASS 3.5.1; the TensorRT-LLM backend was not built.
Baselines: Hugging Face Transformers 5.19 (PyTorch 2.8) and vLLM 0.31.0. vLLM used PyTorch 2.13 cu130 on
the first pod and the cu129 build of the same vLLM release on the second, whose driver predates CUDA 13.
The raw per-run JSON is in this directory, and `quick/REPORT.md` and `sweep/REPORT.md` are generated from
it with `python -m redline.report`.

**Models.** Llama 3.1 8B Instruct (target) and Llama 3.2 1B Instruct (draft). The checkpoints are the
ungated re-uploads `NousResearch/Meta-Llama-3.1-8B-Instruct` and `unsloth/Llama-3.2-1B-Instruct` of Meta's
weights. vLLM INT4 uses `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` (AWQ, Marlin kernels).

| Redline config | weights | lm_head | KV cache | file size |
|---|---|---|---|---:|
| `fp16` | FP16 | FP16 | FP16 | 16.1 GB |
| `int8` | INT8 per-channel (W8A16) | INT8 | FP16 | 8.6 GB |
| `int8_kvint8` | INT8 per-channel | INT8 | INT8 per (token, head) | 8.6 GB |
| `int4` | INT4 g128 asymmetric, clip search (W4A16). `--int4-kernel auto`: CUDA-core GEMV when the engine serves ≤ 4 sequences, tensor-core GEMM above | INT8 | FP16 | 5.3 GB |
| `int4_tc`, `int4_gemv` | `int4` with the kernel forced to tensor cores / to the CUDA-core GEMV | INT8 | FP16 | 5.3 GB |
| `int4_spec` | `int4` target + `int4` 1B draft, k = 4 | INT8 | FP16 | 5.3 + 1.3 GB |
| `int4` AWQ / GPTQ / AWQ+GPTQ | `int4` format, calibrated (quality only: same kernels and speed) | INT8 | FP16 | 5.3 GB |

Every system runs the same workload files: identical prompt token ids and output lengths, with greedy
decoding and EOS ignored. Offline workloads run as successive static batches of `batch_size`, and TTFT is
measured from batch submission. Decode throughput is `batch_size × 1000 / TPOT p50`, where TPOT is a
request's mean time between output tokens.

## Correctness on the GPU

- `redline_cuda_tests`: **24/24 pass** on the first pod, and **29/29** on the second (the tensor-core INT4 GEMM, its split-K and fused SwiGLU epilogue, the fused SwiGLU GEMV, and the GEMV and tensor-core engines agreeing). `pytest tests/python`: 22/22, including AWQ/GPTQ exports against dequantized Transformers logits. This covers every kernel against a CPU reference, all GEMM paths,
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
| `int8` | **99.6%** | **8.108** | 0.704 | 0.796 | 0.516 |
| `int4` (round-to-nearest) | 94.8% | 9.068 | 0.680 | 0.748 | 0.520 |
| `int4` AWQ | 96.1% | 8.502 | 0.684 | 0.756 | 0.508 |
| `int4` GPTQ | **97.1%** | 8.732 | 0.692 | 0.752 | 0.540 |
| `int4` AWQ + GPTQ | 96.4% | **8.468** | 0.688 | 0.772 | 0.500 |

INT4 round-to-nearest g128 costs 11.8% WikiText perplexity on Llama 3 8B, in line with published RTN results
for this quantization-sensitive model family. Calibration fixes most of that. The calibrated files
come from `python -m redline.convert --method awq|gptq|awq_gptq`, which calibrates on 128 C4 samples of 512
tokens, one layer at a time, in 1.5 (AWQ) to 6 (GPTQ) minutes on the 4090. They have the same format and size as
the RTN file, so decode speed is unchanged.

- **Perplexity** is the steadiest signal (256k tokens). AWQ + GPTQ brings the gap to FP16 down from +11.8%
  to +4.4%, which recovers 62% of it. AWQ alone gets to +4.9%. GPTQ alone gets to +7.7%: it fixes rounding
  error inside each layer but cannot move outlier channels, while AWQ's per-channel scales can.
- **Mean retention** goes from 94.8% to 96.1–97.1%. The task scores use 250 examples each (standard error
  about ±3 points), so the order of the calibrated methods within that range is noise. GPTQ's 97.1% is
  helped by an ARC-Challenge score above FP16. AWQ + GPTQ, the best on perplexity and ARC-Easy, is the
  recommended setting.
- The AWQ checkpoint vLLM serves (`hugging-quants/...-AWQ-INT4`) was not run through this evaluation. Its
  quality is not compared here.

## Decode throughput vs batch size (512 prompt / 256 output tokens)

Each cell is TPOT p50 in ms, then decode tok/s. INT4 and vLLM AWQ columns are from the second pod, the
others from the first.

| batch | Redline FP16 | Redline INT8 | Redline INT4 | Redline INT4 + spec* | HF FP16 | vLLM FP16 | vLLM AWQ INT4 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.12 / 58 | 9.27 / 108 | **5.81 / 172** | 3.08 / 325 | 19.83 / 50 | 16.72 / 60 | 6.10 / 164 |
| 2 | 17.31 / 116 | 9.34 / 214 | **5.89 / 339** | 3.12 / 640 | 20.66 / 97 | 16.98 / 118 | 6.25 / 320 |
| 4 | 17.43 / 230 | 9.49 / 422 | **6.11 / 654** | 3.53 / 1133 | 21.09 / 190 | 17.14 / 233 | 6.40 / 625 |
| 8 | 18.20 / 439 | 10.56 / 758 | 7.23 / 1107 | 5.86 / 1366 | 21.96 / 364 | 17.42 / 459 | **6.77 / 1181** |
| 16 | 19.57 / 818 | 18.46 / 867 | 8.85 / 1809 | 17.21 / 930 | 23.85 / 671 | 19.07 / 839 | **8.39 / 1908** |
| 32 | 23.95 / 1336 | 23.32 / 1372 | 14.51 / 2206 | 24.82 / 1289 | 29.28 / 1093 | 22.56 / 1419 | **11.61 / 2755** |

\* Prompts in this sweep are random token ids. The models' continuations of them are repetitive, so draft
acceptance is unrealistically high (0.70–0.84). On WikiText prompts acceptance is 0.40–0.52; see below.

Geometric mean over the six batch sizes:

| | vs HF FP16 | vs vLLM FP16 | vs vLLM AWQ INT4 |
|---|---:|---:|---:|
| Redline FP16 | +20% | −3% | −61% |
| Redline INT8 | **+81%** | +46% | −41% |
| Redline INT4 | **+197%** | **+139%** | −3% |
| Redline INT4, previous kernel (CUDA-core GEMV only) | +124% | +80% | −27% |

## INT4 on tensor cores

The first pod showed INT4 decode flattening at about 950 tok/s from batch 8 up, while vLLM's Marlin
kernel kept scaling. The weight-only GEMV runs on CUDA cores and multiplies every weight by every row,
so past about 8 rows it is limited by FMA and L1 throughput, not DRAM. The second session added a
W4A16 GEMM on tensor cores:

- **Kernel.** `mma.sync.m16n8k16`, fp32 accumulate, 1–64 rows (`src/cuda/kernels/gemm_w4a16_tc.cu`).
- **Weight layout.** INT4 weights are re-laid out once at load (`src/core/int4_layout.cpp`), so each
  lane's 16-byte load is exactly its B fragments for eight n8 tiles.
- **Dequantization.** Nibbles become fp16 with one `lop3` into the mantissa of 1024.0 and a subtract.
  One `hfma2` then applies the group scale and offset.
- **Split-K.** K is split across CTAs and the last CTA of each column block reduces the partials in a
  fixed order. The split count depends only on the layer shape, so results don't depend on the batch.
- **Fused SwiGLU.** The gate/up epilogue writes SiLU(gate)·up directly. The batch-1 GEMV got the same
  fusion through row-interleaved gate/up weights.

The engine picks a layout when it loads the weights. The default (`--int4-kernel auto`) is the GEMV for
engines serving ≤ 4 sequences and tensor cores above that, because one weight layout serves both. The
sweep runs `auto` with `--max-seqs` equal to the batch size, as the suite runner does.

| decode tok/s (TPOT p50 ms) | b1 | b2 | b4 | b8 | b16 | b32 |
|---|---:|---:|---:|---:|---:|---:|
| previous commit, second pod (`control/`) | 170 (5.90) | | | 899 (8.90) | | 938 (34.11) |
| GEMV only (`--int4-kernel gemv`) | 172 (5.82) | 339 | 654 | 897 | 951 | 953 (33.58) |
| tensor cores only (`--int4-kernel tc`) | 161 (6.21) | 319 | 628 | 1,108 | 1,805 | 2,201 (14.54) |
| **auto** | **172 (5.81)** | **339** | **654** | **1,107** | **1,809** | **2,206 (14.51)** |
| vLLM AWQ (Marlin) | 164 (6.10) | 320 | 625 | 1,181 | 1,908 | 2,755 (11.61) |


- At batch 32 INT4 decode is **2.3× faster** than with the CUDA-core GEMV: 2,206 vs 953 tok/s, which is
  80% of vLLM's Marlin kernel, up from 35%. At batch 1–4 it stays 5–6% ahead of vLLM.
- TPOT includes the decode steps that run alongside the batch's chunked prefill, and Redline's prefill
  is 5× slower than vLLM's (TTFT 1.21 s vs 0.24 s at batch 32). Counting steady-state decode steps alone
  (ITL p50), Redline is level with vLLM at batch 8 (6.75 vs 6.78 ms), 3% behind at 16 (7.88 vs 7.68) and
  18% behind at 32 (11.52 vs 9.48).
- At batch 1 the tensor-core path is 7% slower than the GEMV (6.21 vs 5.81 ms), which is why `auto`
  keeps the GEMV for small engines.

Per-op GPU time per forward pass (`redline_bench --profile-ops`, run eagerly, so absolute times are
higher than in graph mode; `profile/`):

| ms per pass | b1 GEMV | b1 tensor cores | b32 GEMV | b32 tensor cores |
|---|---:|---:|---:|---:|
| QKV projection | 0.61 | 0.72 | 3.46 | **1.64** |
| attention (decode) | 0.94 | 0.95 | 4.03 | 4.18 |
| O projection | 0.44 | 0.55 | 2.77 | **1.22** |
| gate/up + SwiGLU | 2.37 | 2.57 | 15.52 | **6.32** |
| down projection | 1.25 | 1.39 | 8.84 | **3.36** |
| lm_head (INT8) | 0.56 | 0.56 | 1.29 | 1.28 |

Tuning, measured on the pod:

- **Split-K reduction.** The last CTA originally summed its partials with one dependent L2 load chain
  per output. Issuing each split's loads together cut batch-32 ITL from 14.75 to 13.51 ms and batch 1
  from 6.49 to 6.33 ms.
- **CTAs per SM.** One CTA per SM was best at every batch size among 0.5, 0.75, 1, 2, 3 and 4 (4- and
  8-warp CTAs). Fewer splits under-fill the GPU. More splits spend longer reducing than they gain in
  memory parallelism. Moving from 2 to 1 took batch 32 from 13.51 to 12.55 ms.
- **INT8 lm_head.** The INT8 lm_head of the INT4 model writes fp32 logits, so it stays on the GEMV.
  Allowing 16 rows per launch instead of 8 halved its passes over the 525 MB projection: 2.28 → 1.28 ms
  at batch 32, and 12.55 → 11.52 ms ITL.

What is left at batch 32: the four GEMMs take 12.5 ms per pass in the eager profile for 3.7 GB of INT4
weights, well short of what DRAM bandwidth allows. Each warp loads its own A fragments from L1, and there
is no shared-memory staging. A Marlin-style kernel stages activations
through shared memory (`cp.async`, `ldmatrix`) and reuses them across more warps. Decode attention (4.2 ms)
is now the second-largest item.

## Quick suite (prompts from WikiText-2)

Decode tok/s on the offline workloads. INT4 and vLLM AWQ rows are from the second pod, on identical
workload files.

| config | b1, 128 in | b1, 1024 in | b8, 128 in | b8, 1024 in | peak VRAM b8/1024 in (GiB) |
|---|---:|---:|---:|---:|---:|
| Redline FP16 | 60 | 58 | 462 | 400 | 16.99 |
| Redline FP16, no graphs, sync | 59 | 57 | 456 | 393 | 16.98 |
| Redline INT8 | 110 | 108 | 830 | 642 | 10.00 |
| Redline INT8 + INT8 KV | 108 | 105 | 818 | 629 | 9.38 |
| Redline INT4 | 177 | 172 | **1278** | 854 | **6.96** |
| Redline INT4 + spec (accept 0.40–0.52) | **205** | **205** | **1319** | 749 | 8.66 |
| HF FP16 | 51 | 50 | 380 | 344 | 18.37 |
| vLLM FP16 | 60 | 60 | 469 | 412 | (pool) |
| vLLM AWQ INT4 | 164 | 163 | 1275 | **925** | (pool) |

Geometric mean over the 4 offline workloads (`quick/REPORT.md`):

| | decode vs HF FP16 | decode vs vLLM FP16 | decode vs vLLM AWQ INT4 | peak VRAM vs HF FP16 |
|---|---:|---:|---:|---:|
| Redline FP16 | +18% | −2% | | −1% |
| Redline INT8 | +109% | +74% | | **−43%** |
| Redline INT8 + INT8 KV | +105% | +71% | | **−45%** |
| Redline INT4 | **+216%** | **+163%** | +1% | −62% |

vLLM preallocates `gpu_memory_utilization × VRAM`, so its peak VRAM is not compared.

**Online, Poisson arrivals at 4 req/s, 50 requests, up to 32 concurrent:**

| system | TTFT p50 / p99 (ms) | TPOT p50 (ms) | output tok/s |
|---|---:|---:|---:|
| Redline FP16 | 69.8 / 195.6 | 21.91 | 603 |
| Redline INT8 | 68.5 / 200.5 | 20.30 | 666 |
| Redline INT4 | 92.8 / 223.4 | 9.32 | 741 |
| Redline INT4 + spec (accept 0.66) | 102.6 / 250.5 | 7.78 | 780 |
| HF FP16 (static batches, no continuous batching) | 1507.9 / 1888.5 | 41.59 | 139 |
| vLLM FP16 | 74.9 / 175.2 | 20.61 | 610 |
| vLLM AWQ INT4 | 51.4 / 148.5 | 7.45 | 745 |

Online INT4 TPOT fell from 23.0 ms (first pod, CUDA-core GEMV) to 9.3 ms: with up to 32 sequences in
flight, most steps now run on tensor cores.

## What the numbers say

1. **Decode is bandwidth-bound, and the batch-1 path is close to the limit.** At batch 1 an FP16 step
   reads 15.0 GB of weights. At 1008 GB/s that is a 14.9 ms floor, and Redline takes 16.8 ms (89% of
   peak bandwidth). INT8 reads 7.5 GB against a 7.45 ms floor and takes 9.1 ms (82%). INT4 reads 4.2 GB
   against a 4.2 ms floor and takes 5.65 ms (74%). The fixed per-step costs (attention, norms, sampling)
   weigh more as weights shrink. Redline INT4 is 5–6% faster than vLLM's Marlin kernel at batch 1–4.
2. **FP16 Redline is on par with vLLM**: 3% slower on average (2% at small batch, 6% at batch 32). It is
   16–22% faster than Transformers.
3. **INT4 needs tensor cores past batch 8, and now has them.** The CUDA-core GEMV stops scaling at about
   950 tok/s. The tensor-core W4A16 GEMM reaches 2,206 tok/s at batch 32: 80% of Marlin, and level with
   it in steady-state decode at batch 8. The rest of the gap is the GEMM's activation staging (above)
   and prefill, which TPOT includes.
4. **Prefill is the weak spot.** At batch 32 × 512 tokens, vLLM's TTFT is 0.24 s against Redline's
   1.2 s. Redline's prefill attention is a tiled CUDA-core kernel, and INT4 prefill dequantizes and
   calls cuBLASLt. A tensor-core (FlashAttention-style) kernel is the next step.
5. **CUDA graphs and async scheduling matter little for an 8B model** (16.80 vs 17.00 ms per step). The
   GPU work per step is far larger than the launch overhead. They matter for the small draft model,
   which runs k rounds per step.
6. **Speculative decoding helps up to batch 8.** On WikiText prompts the 1B draft is accepted 40–52% of
   the time. That gives 1.16–1.19× at batch 1 (205 tok/s) and 1.03× at batch 8 with 128-token prompts, but
   0.88× at batch 8 with 1024-token prompts. Online TPOT drops from 9.3 to 7.8 ms. From batch 16 up, the verification pass has more than 64
   rows and leaves the tensor-core kernel, so speculative decoding is slower than plain INT4.
7. **INT8 KV** halves KV memory (peak −45% vs −43% for INT8 weights alone) at a 1–2% decode cost at these
   context lengths.
8. **Calibration recovers most of INT4's quality loss at no speed cost** (quality section above).

## Reproduce

```bash
scripts/run_rtx4090.sh                                   # full pipeline (needs an RTX 4090 class GPU)
python -m redline.report --results results/rtx4090/quick --baseline hf_fp16 --baseline vllm \
    --baseline vllm_awq_int4 --quality results/rtx4090/quality.json
```
