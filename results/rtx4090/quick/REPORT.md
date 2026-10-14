# Redline benchmark report

## Headline

Offline workloads only; decode throughput = batch x 1000 / TPOT p50. Peak VRAM is not compared against vLLM, which preallocates its memory pool.

| config | vs | workloads | decode speedup (geomean) | range | peak VRAM reduction |
|---|---|---:|---:|---:|---:|
| redline_fp16 | hf_fp16 | 4 | +17.7% | 1.16x-1.21x | 1.3% |
| redline_fp16 | vllm | 4 | -1.9% | 0.97x-0.99x | n/a |
| redline_fp16 | vllm_awq_int4 | 4 | -62.3% | 0.36x-0.43x | n/a |
| redline_fp16_nograph_sync | hf_fp16 | 4 | +16.1% | 1.14x-1.20x | 1.4% |
| redline_fp16_nograph_sync | vllm | 4 | -3.3% | 0.95x-0.98x | n/a |
| redline_fp16_nograph_sync | vllm_awq_int4 | 4 | -62.8% | 0.35x-0.43x | n/a |
| redline_int4 | hf_fp16 | 4 | +181.3% | 2.10x-3.42x | 61.6% |
| redline_int4 | vllm | 4 | +134.4% | 1.76x-2.89x | n/a |
| redline_int4 | vllm_awq_int4 | 4 | -9.8% | 0.77x-1.06x | n/a |
| redline_int4_spec | hf_fp16 | 4 | +176.8% | 1.56x-4.48x | 52.4% |
| redline_int4_spec | vllm | 4 | +130.6% | 1.30x-3.77x | n/a |
| redline_int4_spec | vllm_awq_int4 | 4 | -11.2% | 0.58x-1.39x | n/a |
| redline_int8 | hf_fp16 | 4 | +108.5% | 1.86x-2.18x | 43.3% |
| redline_int8 | vllm | 4 | +73.7% | 1.56x-1.83x | n/a |
| redline_int8 | vllm_awq_int4 | 4 | -33.2% | 0.65x-0.69x | n/a |
| redline_int8_kvint8 | hf_fp16 | 4 | +104.7% | 1.83x-2.15x | 45.4% |
| redline_int8_kvint8 | vllm | 4 | +70.5% | 1.53x-1.80x | n/a |
| redline_int8_kvint8 | vllm_awq_int4 | 4 | -34.4% | 0.64x-0.68x | n/a |

## Evaluation quality (vs FP16 Hugging Face)

| model | retention | per task |
|---|---:|---|
| l8b-int8.rlm | 99.57% | wikitext 100.0%, hellaswag 100.6%, arc_easy 100.0%, arc_challenge 97.7% |
| l8b-int4.rlm | 94.75% | wikitext 89.4%, hellaswag 97.1%, arc_easy 94.0%, arc_challenge 98.5% |

## Per-workload results

#### offline_b1_in1024_out128_greedy

| system | decode tok/s | TTFT p50 / p99 (ms) | TPOT p50 (ms) | ITL p99 (ms) | peak VRAM (GiB) | accept |
|---|---:|---:|---:|---:|---:|---:|
| hf_fp16 | 50.2 | 103.4 / 103.4 | 19.91 | 20.82 | 16.22 | - |
| redline_fp16 | 58.3 | 108.4 / 108.5 | 17.16 | 17.18 | 15.89 | - |
| redline_fp16_nograph_sync | 57.5 | 110.2 / 110.3 | 17.40 | 17.51 | 15.88 | - |
| redline_int4 | 168.4 | 123.9 / 124.1 | 5.94 | 5.98 | 5.85 | - |
| redline_int4_spec | 225.1 | 143.8 / 144.1 | 4.44 | 12.53 | 7.24 | 0.46 |
| redline_int8 | 107.7 | 108.6 / 108.8 | 9.28 | 9.29 | 8.90 | - |
| redline_int8_kvint8 | 105.5 | 109.4 / 109.5 | 9.48 | 9.50 | 8.81 | - |
| vllm | 59.7 | 62.5 / 100.4 | 16.76 | 17.41 | 21.16 | - |
| vllm_awq_int4 | 162.5 | 55.5 / 97.8 | 6.15 | 6.60 | 21.96 | - |

#### offline_b1_in128_out128_greedy

| system | decode tok/s | TTFT p50 / p99 (ms) | TPOT p50 (ms) | ITL p99 (ms) | peak VRAM (GiB) | accept |
|---|---:|---:|---:|---:|---:|---:|
| hf_fp16 | 50.7 | 26.2 / 26.4 | 19.71 | 21.01 | 15.95 | - |
| redline_fp16 | 59.5 | 21.4 / 21.4 | 16.80 | 16.91 | 15.89 | - |
| redline_fp16_nograph_sync | 58.8 | 21.4 / 21.4 | 17.00 | 17.14 | 15.88 | - |
| redline_int4 | 173.6 | 28.7 / 29.0 | 5.76 | 5.87 | 5.85 | - |
| redline_int4_spec | 217.1 | 31.9 / 32.0 | 4.61 | 12.15 | 7.24 | 0.40 |
| redline_int8 | 109.8 | 19.3 / 19.3 | 9.11 | 9.22 | 8.90 | - |
| redline_int8_kvint8 | 108.0 | 19.4 / 19.4 | 9.26 | 9.44 | 8.81 | - |
| vllm | 60.0 | 41.6 / 41.9 | 16.67 | 17.33 | 21.05 | - |
| vllm_awq_int4 | 164.4 | 16.9 / 23.7 | 6.08 | 6.60 | 21.85 | - |

#### offline_b8_in1024_out128_greedy

| system | decode tok/s | TTFT p50 / p99 (ms) | TPOT p50 (ms) | ITL p99 (ms) | peak VRAM (GiB) | accept |
|---|---:|---:|---:|---:|---:|---:|
| hf_fp16 | 344.2 | 987.3 / 987.4 | 23.24 | 24.53 | 18.37 | - |
| redline_fp16 | 399.7 | 637.3 / 879.4 | 20.01 | 25.29 | 16.99 | - |
| redline_fp16_nograph_sync | 393.2 | 643.2 / 895.9 | 20.35 | 34.96 | 16.98 | - |
| redline_int4 | 724.1 | 699.1 / 961.6 | 11.05 | 23.56 | 6.96 | - |
| redline_int4_spec | 536.4 | 675.5 / 1084.2 | 14.91 | 37.33 | 8.65 | 0.51 |
| redline_int8 | 641.8 | 656.6 / 899.4 | 12.46 | 22.81 | 10.00 | - |
| redline_int8_kvint8 | 628.7 | 658.8 / 905.9 | 12.73 | 23.49 | 9.38 | - |
| vllm | 412.2 | 546.5 / 784.7 | 19.41 | 106.63 | 21.30 | - |
| vllm_awq_int4 | 925.0 | 529.0 / 754.6 | 8.65 | 96.08 | 22.12 | - |

#### offline_b8_in128_out128_greedy

| system | decode tok/s | TTFT p50 / p99 (ms) | TPOT p50 (ms) | ITL p99 (ms) | peak VRAM (GiB) | accept |
|---|---:|---:|---:|---:|---:|---:|
| hf_fp16 | 380.2 | 101.5 / 101.8 | 21.04 | 23.04 | 16.24 | - |
| redline_fp16 | 461.9 | 94.0 / 94.8 | 17.32 | 17.45 | 16.99 | - |
| redline_fp16_nograph_sync | 455.8 | 94.7 / 95.6 | 17.55 | 17.75 | 16.98 | - |
| redline_int4 | 986.8 | 110.4 / 111.3 | 8.11 | 8.39 | 6.96 | - |
| redline_int4_spec | 746.8 | 125.4 / 126.0 | 10.71 | 31.20 | 8.65 | 0.49 |
| redline_int8 | 829.7 | 94.0 / 94.2 | 9.64 | 9.77 | 10.00 | - |
| redline_int8_kvint8 | 817.9 | 94.1 / 94.2 | 9.78 | 9.95 | 9.38 | - |
| vllm | 469.0 | 117.1 / 133.1 | 17.06 | 17.74 | 21.14 | - |
| vllm_awq_int4 | 1276.2 | 99.9 / 112.8 | 6.27 | 6.82 | 21.96 | - |

#### online_rate4_greedy

| system | output tok/s | TTFT p50 / p99 (ms) | TPOT p50 (ms) | ITL p99 (ms) | peak VRAM (GiB) | accept |
|---|---:|---:|---:|---:|---:|---:|
| hf_fp16 | 138.7 | 1507.9 / 1888.5 | 41.59 | 56.23 | 23.84 | - |
| redline_fp16 | 602.6 | 69.8 / 195.6 | 21.91 | 76.50 | 20.45 | - |
| redline_fp16_nograph_sync | 602.1 | 50.1 / 184.0 | 22.19 | 76.25 | 20.79 | - |
| redline_int4 | 669.9 | 85.9 / 208.0 | 23.04 | 85.66 | 13.73 | - |
| redline_int4_spec | 751.8 | 72.6 / 232.9 | 15.82 | 157.15 | 17.24 | 0.66 |
| redline_int8 | 665.7 | 68.5 / 200.5 | 20.30 | 79.56 | 16.77 | - |
| redline_int8_kvint8 | 660.2 | 69.5 / 201.3 | 20.50 | 90.52 | 12.88 | - |
| vllm | 610.4 | 74.9 / 175.2 | 20.61 | 65.81 | 21.32 | - |
| vllm_awq_int4 | 746.2 | 42.4 / 150.4 | 7.36 | 31.20 | 22.12 | - |
