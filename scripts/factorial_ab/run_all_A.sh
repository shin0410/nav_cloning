#!/usr/bin/env bash
# A案(事前学習×拡張×データ量×シード)を一括実行し、図表まで作る。
# 中断しても再実行すれば続きから走る(学習済み・評価済みはスキップ)。
#
# 使い方:
#   bash run_all_A.sh              # フル(5エンコーダ×2拡張×4データ量×3シード=120セル)
#   QUICK=1 bash run_all_A.sh      # 短縮版(3エンコーダ×シード1、まず全体を1周させたい時)
#   DEVICE=cuda bash run_all_A.sh  # デバイス指定(省略時は自動)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$SCRIPT_DIR/../../.venv/bin/python"
[[ -x "$PY" ]] || PY=python3

DATA_ROOT="${DATA_ROOT:-/home/shin/challenge_ws/nav_cloning_data}"
TRAIN_TIME="${TRAIN_TIME:-20260308_120428}"           # 12時のデータで学習
TEST_SAME="${TEST_SAME:-20260308_130242}"             # 13時 = ほぼ同条件(1hギャップ)
TEST_GAP="${TEST_GAP:-20260308_100554 20260308_160405 20260310_180327}"  # 2h/4h/6hギャップ
OUT_DIR="${OUT_DIR:-$DATA_ROOT/_factorial_A_$(date +%Y%m%d)}"

if [[ "${QUICK:-0}" == "1" ]]; then
  ENCODERS="${ENCODERS:-scratch,resnet18,dinov2_vits14}"
  SEEDS="${SEEDS:-1}"
else
  ENCODERS="${ENCODERS:-scratch,resnet18,vit_b_16,dinov2_vits14,clip_vitb16}"
  SEEDS="${SEEDS:-1,2,3}"
fi
ROWS_LIST="${ROWS_LIST:-1000,2000,4000,all}"
AUGS="${AUGS:-none,taw3op}"

EXTRA=()
if [[ -n "${DEVICE:-}" ]]; then
  EXTRA+=(--device "$DEVICE")
fi

echo "[START] $(date '+%F %T')  out=$OUT_DIR"
echo "        encoders=$ENCODERS seeds=$SEEDS rows=$ROWS_LIST augs=$AUGS"

"$PY" "$SCRIPT_DIR/run_factorial.py" \
  --data_root "$DATA_ROOT" \
  --train_times "$TRAIN_TIME" \
  --test_times_same "$TEST_SAME" \
  --test_times_gap $TEST_GAP \
  --rows_list "$ROWS_LIST" \
  --encoders "$ENCODERS" \
  --augs "$AUGS" \
  --seeds "$SEEDS" \
  --out_dir "$OUT_DIR" \
  "${EXTRA[@]}"

"$PY" "$SCRIPT_DIR/plot_factorial.py" \
  --results "$OUT_DIR/results_long.csv" \
  --out_dir "$OUT_DIR/plots"

echo "[END] $(date '+%F %T')"
echo "結果:"
echo "  $OUT_DIR/results_long.csv                     … 全セルの生データ"
echo "  $OUT_DIR/plots/01_data_efficiency_same.png    … データ効率曲線(同条件)"
echo "  $OUT_DIR/plots/01_data_efficiency_gap.png     … データ効率曲線(条件ギャップ)"
echo "  $OUT_DIR/plots/02_aug_effect.png              … 拡張×事前学習の交互作用"
echo "  $OUT_DIR/plots/data_requirement.csv           … 必要データ量とデータ削減倍率"
