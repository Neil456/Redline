# Redline benchmark report

## Headline

Offline workloads only; decode throughput = batch x 1000 / TPOT p50. Peak VRAM is not compared against vLLM, which preallocates its memory pool.

| config | vs | workloads | decode speedup (geomean) | range | peak VRAM reduction |
|---|---|---:|---:|---:|---:|
| redline_fp16 | hf_fp16 | 6 | +20.1% | 1.16x-1.22x | -13.5% |
| redline_fp16 | vllm | 6 | -3.1% | 0.94x-0.98x | n/a |
| redline_fp16 | vllm_awq_int4 | 6 | -60.7% | 0.36x-0.48x | n/a |
| redline_int4 | hf_fp16 | 6 | +118.9% | 0.85x-3.44x | 44.1% |
| redline_int4 | vllm | 6 | +76.5% | 0.65x-2.83x | n/a |
| redline_int4 | vllm_awq_int4 | 6 | -28.5% | 0.34x-1.04x | n/a |
| redline_int4_spec | hf_fp16 | 6 | +211.2% | 1.48x-6.83x | 29.8% |
| redline_int4_spec | vllm | 6 | +151.0% | 1.14x-5.76x | n/a |
| redline_int4_spec | vllm_awq_int4 | 6 | +1.7% | 0.59x-2.10x | n/a |
| redline_int8 | hf_fp16 | 6 | +81.3% | 1.26x-2.22x | 26.6% |
| redline_int8 | vllm | 6 | +46.2% | 0.97x-1.82x | n/a |
| redline_int8 | vllm_awq_int4 | 6 | -40.8% | 0.45x-0.67x | n/a |

## Per-workload results

#### sweep_b16_in512_out256

| system | decode tok/s | TTFT p50 / p99 (ms) | TPOT p50 (ms) | ITL p99 (ms) | peak VRAM (GiB) | accept |
|---|---:|---:|---:|---:|---:|---:|
| hf_fp16 | 670.8 | 979.7 / 979.7 | 23.85 | 24.65 | 18.32 | - |
| redline_fp16 | 817.6 | 593.8 / 815.4 | 19.57 | 19.16 | 19.76 | - |
| redline_int4 | 922.7 | 653.6 / 902.2 | 17.34 | 16.89 | 9.72 | - |
| redline_int4_spec | 1250.7 | 626.8 / 1011.4 | 12.79 | 48.10 | 12.22 | 0.74 |
| redline_int8 | 866.6 | 607.3 / 828.4 | 18.46 | 18.07 | 12.77 | - |
| vllm | 839.1 | 222.0 / 428.8 | 19.07 | 19.72 | 22.13 | - |
| vllm_awq_int4 | 1908.4 | 206.3 / 399.2 | 8.38 | 8.75 | 21.34 | - |

#### sweep_b1_in512_out256

| system | decode tok/s | TTFT p50 / p99 (ms) | TPOT p50 (ms) | ITL p99 (ms) | peak VRAM (GiB) | accept |
|---|---:|---:|---:|---:|---:|---:|
| hf_fp16 | 50.4 | 54.3 / 54.3 | 19.83 | 22.01 | 16.07 | - |
| redline_fp16 | 58.4 | 51.6 / 51.6 | 17.12 | 17.15 | 19.76 | - |
| redline_int4 | 168.9 | 67.6 / 67.6 | 5.92 | 5.94 | 9.72 | - |
| redline_int4_spec | 344.3 | 77.8 / 77.8 | 2.90 | 12.46 | 12.22 | 0.82 |
| redline_int8 | 107.8 | 52.2 / 52.2 | 9.27 | 9.28 | 12.77 | - |
| vllm | 59.8 | 22.3 / 22.3 | 16.72 | 17.38 | 21.88 | - |
| vllm_awq_int4 | 164.0 | 14.8 / 14.8 | 6.10 | 6.56 | 21.09 | - |

#### sweep_b2_in512_out256

| system | decode tok/s | TTFT p50 / p99 (ms) | TPOT p50 (ms) | ITL p99 (ms) | peak VRAM (GiB) | accept |
|---|---:|---:|---:|---:|---:|---:|
| hf_fp16 | 96.8 | 102.5 / 102.5 | 20.66 | 22.36 | 16.24 | - |
| redline_fp16 | 115.5 | 99.5 / 99.5 | 17.31 | 17.33 | 19.76 | - |
| redline_int4 | 333.1 | 115.8 / 115.8 | 6.00 | 6.06 | 9.72 | - |
| redline_int4_spec | 531.4 | 134.2 / 134.2 | 3.76 | 16.41 | 12.22 | 0.84 |
| redline_int8 | 214.2 | 99.7 / 99.7 | 9.34 | 9.35 | 12.77 | - |
| vllm | 117.8 | 59.5 / 80.3 | 16.98 | 17.52 | 21.91 | - |
| vllm_awq_int4 | 318.9 | 45.4 / 61.3 | 6.27 | 6.95 | 21.12 | - |

#### sweep_b32_in512_out256

| system | decode tok/s | TTFT p50 / p99 (ms) | TPOT p50 (ms) | ITL p99 (ms) | peak VRAM (GiB) | accept |
|---|---:|---:|---:|---:|---:|---:|
| hf_fp16 | 1093.0 | 1966.6 / 1966.6 | 29.28 | 31.54 | 21.00 | - |
| redline_fp16 | 1336.2 | 1004.4 / 1638.7 | 23.95 | 202.56 | 19.76 | - |
| redline_int4 | 924.6 | 1103.8 / 1804.3 | 34.61 | 221.71 | 9.72 | - |
| redline_int4_spec | 1615.5 | 1136.9 / 2024.9 | 19.81 | 68.55 | 12.22 | 0.71 |
| redline_int8 | 1372.0 | 1018.9 / 1663.7 | 23.32 | 207.30 | 12.77 | - |
| vllm | 1418.5 | 226.9 / 827.7 | 22.56 | 45.95 | 22.13 | - |
| vllm_awq_int4 | 2758.0 | 221.9 / 792.7 | 11.60 | 38.03 | 21.34 | - |

#### sweep_b4_in512_out256

| system | decode tok/s | TTFT p50 / p99 (ms) | TPOT p50 (ms) | ITL p99 (ms) | peak VRAM (GiB) | accept |
|---|---:|---:|---:|---:|---:|---:|
| hf_fp16 | 189.7 | 202.1 / 202.1 | 21.09 | 22.52 | 16.52 | - |
| redline_fp16 | 229.5 | 195.8 / 195.8 | 17.43 | 17.46 | 19.76 | - |
| redline_int4 | 641.9 | 214.7 / 214.7 | 6.23 | 6.27 | 9.72 | - |
| redline_int4_spec | 666.0 | 249.1 / 249.1 | 6.01 | 26.74 | 12.22 | 0.84 |
| redline_int8 | 421.5 | 201.3 / 201.3 | 9.49 | 9.51 | 12.77 | - |
| vllm | 233.4 | 131.2 / 131.3 | 17.14 | 17.75 | 21.97 | - |
| vllm_awq_int4 | 624.6 | 117.0 / 117.1 | 6.40 | 6.82 | 21.17 | - |

#### sweep_b8_in512_out256

| system | decode tok/s | TTFT p50 / p99 (ms) | TPOT p50 (ms) | ITL p99 (ms) | peak VRAM (GiB) | accept |
|---|---:|---:|---:|---:|---:|---:|
| hf_fp16 | 364.3 | 417.0 / 417.0 | 21.96 | 22.31 | 17.19 | - |
| redline_fp16 | 439.5 | 296.1 / 414.7 | 18.20 | 17.92 | 19.76 | - |
| redline_int4 | 882.7 | 324.3 / 446.2 | 9.06 | 8.77 | 9.72 | - |
| redline_int4_spec | 912.5 | 374.7 / 500.4 | 8.77 | 34.22 | 12.22 | 0.72 |
| redline_int8 | 757.8 | 302.7 / 421.9 | 10.56 | 10.22 | 12.77 | - |
| vllm | 459.3 | 221.9 / 238.2 | 17.42 | 18.15 | 22.13 | - |
| vllm_awq_int4 | 1180.8 | 211.1 / 217.5 | 6.78 | 7.49 | 21.34 | - |
