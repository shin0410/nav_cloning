#!/usr/bin/env bash
set -euo pipefail

# ===== ここを必要に応じて編集 =====
WS="$HOME/challenge_ws"
NAV="$WS/src/nav_cloning"
DATA="$NAV/data"
BASE_TRAIN_TIME="20250903_12:05:43"
EVAL_TIMES=("20250903_12:10"
"20250829_20:25:14"
"20250903_12:10:21"
"20260109_12:12:12"
"20260109_14:27:17"
"20260109_15:07:21"
"20260109_15:53:26"
"20260109_16:10:51"
"20260109_17:02:25")  # ←「12時のデータ2つ」を想定

K=3
EPOCHS=100
BATCH=8

# スクリプト実パス（あなたの環境に合わせて調整）
CFG="$NAV/config/config.yaml"
AUGMIX_PY="$NAV/scripts/edit_dataset/aug_mix.py"
TAW_PY="$NAV/scripts/edit_dataset/aug_TrivialAugmentWide.py"
#LEARN_PY="$NAV/scripts/learning_default.py"
LEARN_PY="$NAV/scripts/learning_surprise.py"

# ここで作った補助スクリプト（同じディレクトリに置いた想定）
MERGE_PY="$PWD/merge_dataset.py"
INFER_PY="$PWD/infer_to_csv.py"
EVAL_PY="$PWD/eval_and_plot.py"

OUTDIR="$DATA/_pipeline_out_${BASE_TRAIN_TIME}"
mkdir -p "$OUTDIR"

# ===== config.yaml を安全に一時編集する関数 =====
backup_cfg="$OUTDIR/config.yaml.bak"
cp -f "$CFG" "$backup_cfg"

set_cfg () {
python3 - <<PY 1>&2
import yaml
p="${CFG}"
d=yaml.safe_load(open(p,'r'))
# 必要キーが無いときは作る
d["time"]="${1}"
d["batch_size"]=str(${BATCH})
d["epoch"]=str(${EPOCHS})
# 学習データ指定
d["load_dataset_img"]="${2}"
d["load_dataset_vel"]="${3}"
# 増強 入出力
d["input_aug_dataset_img"]="img"
d["input_aug_dataset_vel"]="vel"
d["output_aug_dataset_img"]="${4}"
d["output_aug_dataset_vel"]="${5}"
# 学習モデル名
d["save_model"]="${6}"
d["load_model"]="${6}"
yaml.safe_dump(d, open(p,'w'), sort_keys=False, allow_unicode=True)
print("[CFG]", p)
PY
}

restore_cfg () {
  cp -f "$backup_cfg" "$CFG"
  echo "[CFG] restored"
}

trap restore_cfg EXIT

# ===== 元データのパス =====
TRAIN_BASE="$DATA/$BASE_TRAIN_TIME/dataset"
ORIG_IMG="$TRAIN_BASE/img"
ORIG_CSV="$TRAIN_BASE/vel/data.csv"

if [[ ! -d "$ORIG_IMG" ]]; then
  echo "[ERR] missing: $ORIG_IMG" >&2; exit 1
fi
if [[ ! -f "$ORIG_CSV" ]]; then
  echo "[ERR] missing: $ORIG_CSV" >&2; exit 1
fi

# ===== 手法ごとの学習データ（merged）を作る =====
# baseline（増強なし）
mk_baseline () {
  local EXP_TIME="${BASE_TRAIN_TIME}__baseline"
  local EXP_DS="$DATA/$EXP_TIME/dataset"
  mkdir -p "$EXP_DS/img" "$EXP_DS/vel"
  python3 "$MERGE_PY" \
    --orig_img "$ORIG_IMG" \
    --orig_csv "$ORIG_CSV" \
    --out_img "$EXP_DS/img" \
    --out_csv "$EXP_DS/vel/data.csv" \
    --mode symlink
  echo "$EXP_TIME"
}

# augmix（増強→merged）
mk_augmix () {
  local EXP_TIME="${BASE_TRAIN_TIME}__augmixK${K}"
  # 1) まず BASE_TRAIN_TIME に augmix_img / augmix_vel を生成（増強コードはいじらない） :contentReference[oaicite:11]{index=11}
  set_cfg "$BASE_TRAIN_TIME" "img" "vel" "augmix_img_K${K}" "augmix_vel_K${K}" "model_gpu_augmix.pt"
  python3 "$AUGMIX_PY" --K "$K"
  local AUG_IMG="$TRAIN_BASE/augmix_img_K${K}"
  local AUG_CSV="$TRAIN_BASE/augmix_vel_K${K}/data.csv"


  # 2) EXP_TIME に merged データを作る
  local EXP_DS="$DATA/$EXP_TIME/dataset"
  mkdir -p "$EXP_DS/img" "$EXP_DS/vel"
  python3 "$MERGE_PY" \
    --orig_img "$ORIG_IMG" \
    --orig_csv "$ORIG_CSV" \
    --aug_img  "$AUG_IMG" \
    --aug_csv  "$AUG_CSV" \
    --out_img "$EXP_DS/img" \
    --out_csv "$EXP_DS/vel/data.csv" \
    --mode symlink 1>&2
  echo "$EXP_TIME"
}

# trivialAugWide no-geom（増強→merged）
mk_taw () {
  local EXP_TIME="${BASE_TRAIN_TIME}__tawK${K}"
  # output名は config.yaml のコメントに合わせておく :contentReference[oaicite:12]{index=12}
  set_cfg "$BASE_TRAIN_TIME" "img" "vel" "TrivialAugmentWide_img_K${K}" "TrivialAugmentWide_vel_K${K}" "model_gpu_taw.pt"
  python3 "$TAW_PY" --K "$K"
  local AUG_IMG="$TRAIN_BASE/TrivialAugmentWide_img_K${K}"
  local AUG_CSV="$TRAIN_BASE/TrivialAugmentWide_vel_K${K}/data.csv"


  local EXP_DS="$DATA/$EXP_TIME/dataset"
  mkdir -p "$EXP_DS/img" "$EXP_DS/vel"
  python3 "$MERGE_PY" \
    --orig_img "$ORIG_IMG" \
    --orig_csv "$ORIG_CSV" \
    --aug_img  "$AUG_IMG" \
    --aug_csv  "$AUG_CSV" \
    --out_img "$EXP_DS/img" \
    --out_csv "$EXP_DS/vel/data.csv" \
    --mode symlink
  echo "$EXP_TIME"
}

# ===== 学習 =====
train_one () {
  local EXP_TIME="$1"
  local SAVE_MODEL="$2"
  set_cfg "$EXP_TIME" "img" "vel" "dummy" "dummy" "$SAVE_MODEL"
  python3 "$LEARN_PY"
}

# ===== 推論＆評価（各eval_timeにpred列を作り、Fig5.9含むPNGを保存） =====
infer_and_eval () {
  local EXP_TIME="$1"
  local SAVE_MODEL="$2"
  local METHOD="$3"

  local MODEL_PATH="$DATA/$EXP_TIME/model/$EPOCHS/$SAVE_MODEL"
  if [[ ! -f "$MODEL_PATH" ]]; then
    echo "[ERR] missing model: $MODEL_PATH" >&2; exit 1
  fi

  for T in "${EVAL_TIMES[@]}"; do
    local DS="$DATA/$T/dataset"
    local IMG="$DS/img"
    local CSV=""
    if [[ -f "$DS/vel/deta.csv" ]]; then
      CSV="$DS/vel/deta.csv"
    else
      CSV="$DS/vel/data.csv"
    fi

    if [[ ! -d "$IMG" || ! -f "$CSV" ]]; then
      echo "[WARN] skip eval time=$T (missing img/csv)" >&2
      continue
    fi

    local RUN_OUT="$OUTDIR/${METHOD}/eval_${T}"
    mkdir -p "$RUN_OUT"

    local PRED_COL="pred_${METHOD}"
    local CSV_OUT="$RUN_OUT/with_pred.csv"

    python3 "$INFER_PY" \
      --img_dir "$IMG" \
      --csv_in "$CSV" \
      --model "$MODEL_PATH" \
      --pred_col "$PRED_COL" \
      --csv_out "$CSV_OUT" \
      --view center

    python3 "$EVAL_PY" \
      --csv "$CSV_OUT" \
      --pred "$PRED_COL" \
      --teacher center \
      --xcol episode \
      --title "${METHOD} on ${T}" \
      --outdir "$RUN_OUT/plots" \
      --json_out "$RUN_OUT/metrics.json"
  done
}

# ===== summary.csv を作る =====
make_summary () {
  python3 - <<PY
import os, json, csv, glob
out="${OUTDIR}"
rows=[]
for mdir in glob.glob(os.path.join(out,"*","eval_*","metrics.json")):
    method = mdir.split(os.sep)[-3]
    evaltime = mdir.split(os.sep)[-2].replace("eval_","")
    met = json.load(open(mdir,"r",encoding="utf-8"))
    met.update({"method": method, "eval_time": evaltime})
    rows.append(met)

rows.sort(key=lambda r:(r["method"], r["eval_time"]))
if not rows:
    print("[WARN] no metrics.json found")
    raise SystemExit(0)

keys=["method","eval_time","count","mae","median","p80","p95","sign_flip_rate","thr_disagree@0.1","thr_disagree@0.2"]
p=os.path.join(out,"summary.csv")
with open(p,"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for r in rows:
        w.writerow({k:r.get(k,"") for k in keys})
print("[DONE] summary:", p)
PY
}

# ===== 実行（baseline + augmix + taw） =====
BASELINE_TIME="$(mk_baseline | tail -n 1)"
AUGMIX_TIME="$(mk_augmix | tail -n 1)"
TAW_TIME="$(mk_taw | tail -n 1)"

train_one "$BASELINE_TIME" "model_gpu_baseline.pt"
train_one "$AUGMIX_TIME"   "model_gpu_augmix.pt"
train_one "$TAW_TIME"      "model_gpu_taw.pt"

infer_and_eval "$BASELINE_TIME" "model_gpu_baseline.pt" "baseline"
infer_and_eval "$AUGMIX_TIME"   "model_gpu_augmix.pt"   "augmix"
infer_and_eval "$TAW_TIME"      "model_gpu_taw.pt"      "taw"

make_summary

echo "=== DONE ==="
echo "OUT: $OUTDIR"

