# Redline benchmark report

## Headline

Offline workloads only; decode throughput = batch x 1000 / TPOT p50. Peak VRAM is not compared against vLLM, which preallocates its memory pool.

| config | vs | workloads | decode speedup (geomean) | range | peak VRAM reduction |
|---|---|---:|---:|---:|---:|
| redline_fp16 | hf_fp16 | 6 | +20.1% | 1.16x-1.22x | -13.5% |
| redline_fp16 | vllm | 6 | -3.1% | 0.94x-0.98x | n/a |
| redline_fp16 | vllm_awq_int4 | 6 | -60.8% | 0.36x-0.48x | n/a |
| redline_int4 | hf_fp16 | 6 | +196.8% | 2.02x-3.51x | 44.2% |
| redline_int4 | vllm | 6 | +139.3% | 1.56x-2.88x | n/a |
| redline_int4 | vllm_awq_int4 | 6 | -3.1% | 0.80x-1.06x | n/a |
| redline_int4_gemv | hf_fp16 | 6 | +123.7% | 0.87x-3.51x | 44.2% |
| redline_int4_gemv | vllm | 6 | +80.4% | 0.67x-2.88x | n/a |
| redline_int4_gemv | vllm_awq_int4 | 6 | -26.9% | 0.35x-1.06x | n/a |
| redline_int4_spec | hf_fp16 | 6 | +240.6% | 1.18x-6.61x | 29.7% |
| redline_int4_spec | vllm | 6 | +174.7% | 0.91x-5.44x | n/a |
| redline_int4_spec | vllm_awq_int4 | 6 | +11.3% | 0.47x-2.00x | n/a |
| redline_int4_tc | hf_fp16 | 6 | +188.3% | 2.01x-3.31x | 44.1% |
| redline_int4_tc | vllm | 6 | +132.5% | 1.55x-2.70x | n/a |
| redline_int4_tc | vllm_awq_int4 | 6 | -5.8% | 0.80x-1.01x | n/a |
| redline_int8 | hf_fp16 | 6 | +81.3% | 1.26x-2.22x | 26.6% |
| redline_int8 | vllm | 6 | +46.2% | 0.97x-1.82x | n/a |
| redline_int8 | vllm_awq_int4 | 6 | -40.8% | 0.45x-0.67x | n/a |

## Evaluation quality (vs FP16 Hugging Face)

| model | retention | per task |
|---|---:|---|
| l8b-int8.rlm | 99.57% | wikitext 100.0%, hellaswag 100.6%, arc_easy 100.0%, arc_challenge 97.7% |
| l8b-int4.rlm | 94.75% | wikitext 89.4%, hellaswag 97.1%, arc_easy 94.0%, arc_challenge 98.5% |
| l8b-int4-awq.rlm | 96.07% | wikitext 95.4%, hellaswag 97.7%, arc_easy 95.0%, arc_challenge 96.2% |
| l8b-int4-gptq.rlm | 97.11% | wikitext 92.9%, hellaswag 98.9%, arc_easy 94.5%, arc_challenge 102.3% |
| l8b-int4-awq_gptq.rlm | 96.43% | wikitext 95.7%, hellaswag 98.3%, arc_easy 97.0%, arc_challenge 94.7% |

## Per-workload results

#### sweep_b16_in512_out256

| system | decode tok/s | TTFT p50 / p99 (ms) | TPOT p50 (ms) | ITL p99 (ms) | peak VRAM (GiB) | accept |
|---|---:|---:|---:|---:|---:|---:|
| hf_fp16 | 670.8 | 979.7 / 979.7 | 23.85 | 24.65 | 18.32 | - |
| redline_fp16 | 817.6 | 593.8 / 815.4 | 19.57 | 19.16 | 19.76 | - |
| redline_int4 | 1808.6 | 717.1 / 973.0 | 8.85 | 11.34 | 9.72 | - |
| redline_int4_gemv | 950.9 | 653.7 / 898.4 | 16.83 | 16.44 | 9.72 | - |
| redline_int4_spec | 929.8 | 688.9 / 1106.9 | 17.21 | 71.66 | 12.23 | 0.74 |
| redline_int4_tc | 1805.1 | 719.4 / 976.1 | 8.86 | 8.27 | 9.73 | - |
| redline_int8 | 866.6 | 607.3 / 828.4 | 18.46 | 18.07 | 12.77 | - |
| vllm | 839.1 | 222.0 / 428.8 | 19.07 | 19.72 | 22.13 | - |
| vllm_awq_int4 | 1907.6 | 218.0 / 407.9 | 8.39 | 8.87 | 22.14 | - |

#### sweep_b1_in512_out256

| system | decode tok/s | TTFT p50 / p99 (ms) | TPOT p50 (ms) | ITL p99 (ms) | peak VRAM (GiB) | accept |
|---|---:|---:|---:|---:|---:|---:|
| hf_fp16 | 50.4 | 54.3 / 54.3 | 19.83 | 22.01 | 16.07 | - |
| redline_fp16 | 58.4 | 51.6 / 51.6 | 17.12 | 17.15 | 19.76 | - |
| redline_int4 | 172.2 | 66.9 / 66.9 | 5.81 | 5.82 | 9.69 | - |
| redline_int4_gemv | 171.9 | 67.3 / 67.3 | 5.82 | 5.84 | 9.72 | - |
| redline_int4_spec | 325.1 | 110.2 / 110.2 | 3.08 | 13.19 | 12.23 | 0.82 |
| redline_int4_tc | 161.1 | 95.3 / 95.3 | 6.21 | 6.22 | 9.73 | - |
| redline_int8 | 107.8 | 52.2 / 52.2 | 9.27 | 9.28 | 12.77 | - |
| vllm | 59.8 | 22.3 / 22.3 | 16.72 | 17.38 | 21.88 | - |
| vllm_awq_int4 | 164.0 | 25.4 / 25.4 | 6.10 | 7.94 | 21.89 | - |

#### sweep_b2_in512_out256

| system | decode tok/s | TTFT p50 / p99 (ms) | TPOT p50 (ms) | ITL p99 (ms) | peak VRAM (GiB) | accept |
|---|---:|---:|---:|---:|---:|---:|
| hf_fp16 | 96.8 | 102.5 / 102.5 | 20.66 | 22.36 | 16.24 | - |
| redline_fp16 | 115.5 | 99.5 / 99.5 | 17.31 | 17.33 | 19.76 | - |
| redline_int4 | 339.3 | 114.4 / 114.4 | 5.89 | 5.92 | 9.69 | - |
| redline_int4_gemv | 339.3 | 115.0 / 115.0 | 5.90 | 5.92 | 9.72 | - |
| redline_int4_spec | 640.2 | 163.4 / 163.4 | 3.12 | 13.60 | 12.23 | 0.84 |
| redline_int4_tc | 318.7 | 142.3 / 142.3 | 6.28 | 6.29 | 9.73 | - |
| redline_int8 | 214.2 | 99.7 / 99.7 | 9.34 | 9.35 | 12.77 | - |
| vllm | 117.8 | 59.5 / 80.3 | 16.98 | 17.52 | 21.91 | - |
| vllm_awq_int4 | 319.8 | 66.3 / 75.1 | 6.25 | 6.67 | 21.92 | - |

#### sweep_b32_in512_out256

| system | decode tok/s | TTFT p50 / p99 (ms) | TPOT p50 (ms) | ITL p99 (ms) | peak VRAM (GiB) | accept |
|---|---:|---:|---:|---:|---:|---:|
| hf_fp16 | 1093.0 | 1966.6 / 1966.6 | 29.28 | 31.54 | 21.00 | - |
| redline_fp16 | 1336.2 | 1004.4 / 1638.7 | 23.95 | 202.56 | 19.76 | - |
| redline_int4 | 2205.9 | 1206.5 / 1999.4 | 14.51 | 243.30 | 9.73 | - |
| redline_int4_gemv | 953.1 | 1096.4 / 1789.1 | 33.58 | 219.99 | 9.72 | - |
| redline_int4_spec | 1289.1 | 1246.5 / 2221.3 | 24.82 | 88.92 | 12.23 | 0.70 |
| redline_int4_tc | 2201.1 | 1208.8 / 2004.1 | 14.54 | 243.53 | 9.73 | - |
| redline_int8 | 1372.0 | 1018.9 / 1663.7 | 23.32 | 207.30 | 12.77 | - |
| vllm | 1418.5 | 226.9 / 827.7 | 22.56 | 45.95 | 22.13 | - |
| vllm_awq_int4 | 2755.4 | 243.0 / 811.8 | 11.61 | 36.76 | 22.14 | - |

#### sweep_b4_in512_out256

| system | decode tok/s | TTFT p50 / p99 (ms) | TPOT p50 (ms) | ITL p99 (ms) | peak VRAM (GiB) | accept |
|---|---:|---:|---:|---:|---:|---:|
| hf_fp16 | 189.7 | 202.1 / 202.1 | 21.09 | 22.52 | 16.52 | - |
| redline_fp16 | 229.5 | 195.8 / 195.8 | 17.43 | 17.46 | 19.76 | - |
| redline_int4 | 654.3 | 211.8 / 211.8 | 6.11 | 6.14 | 9.70 | - |
| redline_int4_gemv | 653.6 | 214.1 / 214.1 | 6.12 | 6.15 | 9.72 | - |
| redline_int4_spec | 1132.8 | 274.5 / 274.5 | 3.53 | 15.49 | 12.23 | 0.84 |
| redline_int4_tc | 628.4 | 238.0 / 238.0 | 6.37 | 6.39 | 9.73 | - |
| redline_int8 | 421.5 | 201.3 / 201.3 | 9.49 | 9.51 | 12.77 | - |
| vllm | 233.4 | 131.2 / 131.3 | 17.14 | 17.75 | 21.97 | - |
| vllm_awq_int4 | 624.7 | 126.0 / 126.1 | 6.40 | 6.95 | 21.97 | - |

#### sweep_b8_in512_out256

| system | decode tok/s | TTFT p50 / p99 (ms) | TPOT p50 (ms) | ITL p99 (ms) | peak VRAM (GiB) | accept |
|---|---:|---:|---:|---:|---:|---:|
| hf_fp16 | 364.3 | 417.0 / 417.0 | 21.96 | 22.31 | 17.19 | - |
| redline_fp16 | 439.5 | 296.1 / 414.7 | 18.20 | 17.92 | 19.76 | - |
| redline_int4 | 1107.2 | 357.1 / 486.2 | 7.23 | 6.83 | 9.71 | - |
| redline_int4_gemv | 897.2 | 322.4 / 443.8 | 8.92 | 8.62 | 9.72 | - |
| redline_int4_spec | 1366.2 | 412.4 / 550.3 | 5.86 | 20.38 | 12.23 | 0.72 |
| redline_int4_tc | 1107.7 | 358.3 / 487.6 | 7.22 | 6.83 | 9.73 | - |
| redline_int8 | 757.8 | 302.7 / 421.9 | 10.56 | 10.22 | 12.77 | - |
| vllm | 459.3 | 221.9 / 238.2 | 17.42 | 18.15 | 22.13 | - |
| vllm_awq_int4 | 1181.2 | 216.3 / 221.3 | 6.77 | 7.37 | 22.14 | - |
