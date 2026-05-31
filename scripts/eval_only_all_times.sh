#!/usr/bin/env bash
set -euo pipefail

NAV_DIR="$HOME/challenge_ws/src/nav_cloning"
DATA_BASE="$NAV_DIR/navcloning_pipeline_runall/data"
CFG="$NAV_DIR/config/config.yaml"

# eval_center.py の場所を自動検出
EVAL_SCRIPT=""
for cand in \
  "$NAV_DIR/navcloning_pipeline_runall/pipeline_scripts/eval_center.py" \
  "$NAV_DIR/navcloning_pipeline_runall/pipeline_scripts/scripts/eval_center.py" \
  "$NAV_DIR/navcloning_pipeline_runall/scripts/eval_center.py"
do
  if [ -f "$cand" ]; then EVAL_SCRIPT="$cand"; break; fi
done
[ -n "$EVAL_SCRIPT" ] || { echo "[ERR] eval_center.py が見つからない"; exit 1; }
echo "[EVAL_SCRIPT] $EVAL_SCRIPT"

# 出力先（新規で作る：既存OUTに上書きしたくないため）
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="$DATA_BASE/_pipeline_out_evalonly_20260112_13:06:37_${STAMP}"
mkdir -p "$OUT_ROOT/eval"
echo "[OUT_ROOT] $OUT_ROOT"

# 使うモデル（学習済みが存在する前提）
MODEL_baseline="20260112_13:06:37__baseline"
MODEL_taw="20260112_13:06:37__tawK1"
MODEL_randaug="20260112_13:06:37__randaugK1_N2M9"
MODEL_augmix="20260112_13:06:37__augmixK1"
# augmix_cons は作らない

TIMES=(
  20260112_09:49:31 20260112_10:01:45 20260112_10:05:46 20260112_10:09:03
  20260112_11:02:15 20260112_11:09:30 20260112_12:03:40 20260112_12:06:28
  20260112_12:09:29 20260112_13:09:38 20260112_13:15:50 20260112_14:08:03
  20260112_14:11:47 20260112_15:03:18 20260112_15:07:57 20260112_16:15:51
  20260112_16:19:24 20260112_17:08:12 20260112_17:11:48 20260112_17:15:12
  20260112_18:10:41 20260112_18:14:00
)

METHODS=(baseline taw randaug augmix)

for T in "${TIMES[@]}"; do
  for M in "${METHODS[@]}"; do
    MT_VAR="MODEL_${M}"
    MT="${!MT_VAR}"
    OD="$OUT_ROOT/eval/$T/$M"
    mkdir -p "$OD"
    echo "[RUN] eval_time=$T method=$M model_time=$MT"
    python "$EVAL_SCRIPT" --config "$CFG" --nav_dir "$NAV_DIR" \
      --model_time "$MT" --eval_time "$T" --out_dir "$OD" --view center
  done
done

echo "[DONE] eval generated under: $OUT_ROOT/eval"
