#!/usr/bin/env bash
# End-to-end Redline evaluation on a single GPU (sized for an RTX 4090, 24 GB):
#   1. build  2. convert FP16 / INT8 / INT4 checkpoints (+ INT4 draft model)
#   3. generate the workload suite  4. run Redline configs, HF and vLLM baselines
#   5. evaluate quality retention   6. write REPORT.md
#
# Usage:  scripts/run_rtx4090.sh                 # full suite (several hours)
#         QUICK=1 scripts/run_rtx4090.sh         # small grid smoke run
# Env:    MODEL, DRAFT, OUT, BUILD, SKIP_VLLM=1, SKIP_HF=1, CUTLASS=1, EVAL_LIMIT
set -euo pipefail

MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
DRAFT=${DRAFT:-meta-llama/Llama-3.2-1B-Instruct}
OUT=${OUT:-runs/$(date +%Y%m%d_%H%M%S)}
BUILD=${BUILD:-build}
QUICK=${QUICK:-0}
EVAL_LIMIT=${EVAL_LIMIT:-1000}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$ROOT/python:${PYTHONPATH:-}"
export REDLINE_BUILD_DIR="$ROOT/$BUILD"

mkdir -p "$OUT/models" "$OUT/results"
echo "== Redline run -> $OUT (model=$MODEL draft=$DRAFT quick=$QUICK)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee "$OUT/gpu.txt"

echo "== 1. build"
cmake -S "$ROOT" -B "$ROOT/$BUILD" -DCMAKE_BUILD_TYPE=Release -DREDLINE_BUILD_PYTHON=ON \
  -DREDLINE_WITH_CUTLASS=$([ "${CUTLASS:-1}" = 1 ] && echo ON || echo OFF)
cmake --build "$ROOT/$BUILD" -j"$(nproc)"
"$ROOT/$BUILD/redline_cuda_tests" | tail -n 3

echo "== 2. convert"
convert() { [ -f "$2" ] || python -m redline.convert --model "$1" --out "$2" "${@:3}"; }
convert "$MODEL" "$OUT/models/target-fp16.rlm" --weights fp16
convert "$MODEL" "$OUT/models/target-int8.rlm" --weights int8 --lm-head int8
convert "$MODEL" "$OUT/models/target-int4.rlm" --weights int4 --group-size 128 --lm-head int8
convert "$DRAFT" "$OUT/models/draft-int4.rlm" --weights int4 --group-size 128 --lm-head int8

echo "== 3. workloads"
python -m redline.workloads suite --tokenizer "$MODEL" --out-dir "$OUT/workloads" $([ "$QUICK" = 1 ] && echo --quick)

echo "== 4. benchmarks"
BASELINES=()
[ "${SKIP_HF:-0}" = 1 ] || BASELINES+=(--hf "$MODEL")
[ "${SKIP_VLLM:-0}" = 1 ] || BASELINES+=(--vllm "$MODEL")
M="$OUT/models"
python -m redline.suite --workloads "$OUT/workloads" --out "$OUT/results" \
  --redline "fp16=$M/target-fp16.rlm" \
  --redline "int8=$M/target-int8.rlm" \
  --redline "int4=$M/target-int4.rlm" \
  --redline "int8_kvint8=$M/target-int8.rlm --kv-dtype int8" \
  --redline "int4_spec=$M/target-int4.rlm --draft-model $M/draft-int4.rlm --spec-tokens 4" \
  --redline "fp16_nograph_sync=$M/target-fp16.rlm --no-cuda-graphs --no-async" \
  "${BASELINES[@]}"

echo "== 5. quality"
python -m redline.eval_quality --hf "$MODEL" --redline "$M/target-int8.rlm" --redline "$M/target-int4.rlm" \
  --limit "$EVAL_LIMIT" --hf-cache "$OUT/quality_hf.json" --out "$OUT/quality.json"

echo "== 6. report"
python -m redline.report --results "$OUT/results" --baseline hf_fp16 $([ "${SKIP_VLLM:-0}" = 1 ] || echo --baseline vllm) \
  --quality "$OUT/quality.json" --out "$OUT/REPORT.md" --json-out "$OUT/summary.json"
echo "done: $OUT/REPORT.md"
