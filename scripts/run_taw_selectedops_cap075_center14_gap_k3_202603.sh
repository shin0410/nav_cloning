#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$NAV_DIR/data"
CFG="$NAV_DIR/config/config.yaml"

SPLIT_PY="$NAV_DIR/scripts/split_dataset_first_n.py"
TAW_PY="$NAV_DIR/scripts/edit_dataset/aug_TrivialAugmentWide.py"
LEARN_PY="$NAV_DIR/scripts/learning_surprise.py"
COMPARE_PY="$NAV_DIR/scripts/compare_models_on_test_times.py"

PYTHON_BIN="${PYTHON_BIN:-python3}"

TRAIN_ROWS=4000
TRAIN_IMG_DIR="train4000_img"
TRAIN_VEL_DIR="train4000_vel"
TEST_IMG_DIR="test_rest_img"
TEST_VEL_DIR="test_rest_vel"
OUTPUT_TIME="_center14_gap_taw4ops_cap075_k3_20260308_0310"
OUT_DIR=""
RUN_UNZIP=1
RUN_SPLIT=1
RUN_TRAIN=1
RUN_EVAL=1

CENTER_TIME="20260308_140206"
K=3
MAG=31
MAX_SEVERITY_RATIO="0.75"
OPS_CSV="Solarize,Sharpness,Color,Brightness"

ALL_TIMES=(
  "20260308_100554"
  "20260308_111130"
  "20260308_120428"
  "20260308_130242"
  "20260308_140206"
  "20260308_150431"
  "20260308_160405"
  "20260310_170214"
  "20260310_180327"
)

# Same clean-times setting as center14_gap K3:
# all times except the 13:00 and 15:00 gap slots, plus the center-time augmentation.
CLEAN_TIMES_K3=(
  "20260308_100554"
  "20260308_111130"
  "20260308_120428"
  "20260308_140206"
  "20260308_160405"
  "20260310_170214"
  "20260310_180327"
)

usage() {
  cat <<'EOF'
Usage:
  run_taw_selectedops_cap075_center14_gap_k3_202603.sh [options]

Options:
  --train-rows N              default: 4000
  --output-time NAME          default: _center14_gap_taw4ops_cap075_k3_20260308_0310
  --out-dir DIR               default: data/<output-time>
  --mag N                     TrivialAugmentWide num-magnitude-bins (default: 31)
  --max-severity-ratio R      default: 0.75
  --ops LIST                  default: Solarize,Sharpness,Color,Brightness
  --skip-unzip                do not extract zip files
  --skip-split                do not rebuild train4000 / test_rest splits
  --skip-train                skip augmentation/training and evaluate existing model
  --skip-eval                 train only
  -h, --help

Scenario:
  center14_gap + TrivialAugmentWide K=3
  allowed ops = Solarize,Sharpness,Color,Brightness
  max severity ratio = 0.75
  center augmentation time = 20260308_140206
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-rows)
      TRAIN_ROWS="${2:?missing value for --train-rows}"
      shift 2
      ;;
    --output-time)
      OUTPUT_TIME="${2:?missing value for --output-time}"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="${2:?missing value for --out-dir}"
      shift 2
      ;;
    --mag)
      MAG="${2:?missing value for --mag}"
      shift 2
      ;;
    --max-severity-ratio)
      MAX_SEVERITY_RATIO="${2:?missing value for --max-severity-ratio}"
      shift 2
      ;;
    --ops)
      OPS_CSV="${2:?missing value for --ops}"
      shift 2
      ;;
    --skip-unzip)
      RUN_UNZIP=0
      shift
      ;;
    --skip-split)
      RUN_SPLIT=0
      shift
      ;;
    --skip-train)
      RUN_TRAIN=0
      shift
      ;;
    --skip-eval)
      RUN_EVAL=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERR] unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

for f in "$CFG" "$SPLIT_PY" "$TAW_PY" "$LEARN_PY" "$COMPARE_PY"; do
  [[ -f "$f" ]] || { echo "[ERR] missing file: $f" >&2; exit 1; }
done

[[ "$MAG" =~ ^[0-9]+$ ]] || { echo "[ERR] --mag must be integer: $MAG" >&2; exit 1; }
(( MAG >= 2 )) || { echo "[ERR] --mag must be >= 2" >&2; exit 1; }

RATIO_CHECK="$MAX_SEVERITY_RATIO" "$PYTHON_BIN" - <<'PY'
import os
r = float(os.environ["RATIO_CHECK"])
if not 0.0 <= r <= 1.0:
    raise SystemExit("[ERR] --max-severity-ratio must be in [0.0, 1.0]")
PY

parse_ops() {
  local csv="$1"
  local -a parsed=()
  declare -A seen=()

  IFS=',' read -r -a tokens <<< "$csv"
  for token in "${tokens[@]}"; do
    local op="${token#"${token%%[![:space:]]*}"}"
    op="${op%"${op##*[![:space:]]}"}"
    [[ -n "$op" ]] || continue
    if [[ -z "${seen[$op]+x}" ]]; then
      seen[$op]=1
      parsed+=("$op")
    fi
  done

  [[ ${#parsed[@]} -gt 0 ]] || { echo "[ERR] no valid --ops values" >&2; exit 1; }
  OPS=("${parsed[@]}")
  OPS_CSV="$(IFS=,; echo "${OPS[*]}")"
}

parse_ops "$OPS_CSV"

AVAILABLE_OPS_RAW="$("$PYTHON_BIN" "$TAW_PY" --num-magnitude-bins "$MAG" --list-ops)"
declare -A AVAIL=()
IFS=',' read -r -a _avail_tokens <<< "$AVAILABLE_OPS_RAW"
for op in "${_avail_tokens[@]}"; do
  AVAIL["$op"]=1
done
for op in "${OPS[@]}"; do
  [[ -n "${AVAIL[$op]+x}" ]] || {
    echo "[ERR] unsupported op '$op' for num-magnitude-bins=$MAG" >&2
    echo "      available: $AVAILABLE_OPS_RAW" >&2
    exit 1
  }
done

safe_ratio_name() {
  RATIO="$MAX_SEVERITY_RATIO" "$PYTHON_BIN" - <<'PY'
import os
r = float(os.environ["RATIO"])
print(f"R{int(round(r * 1000)):04d}")
PY
}

ops_safe_name() {
  OPS_RAW="$OPS_CSV" "$PYTHON_BIN" - <<'PY'
import os
import re
ops = [x.strip() for x in os.environ["OPS_RAW"].split(",") if x.strip()]
print("OPS" + "_".join(re.sub(r"[^A-Za-z0-9_]+", "_", op) for op in ops))
PY
}

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$DATA_DIR/$OUTPUT_TIME"
fi

SAFE_RATIO="$(safe_ratio_name)"
SAFE_OPS="$(ops_safe_name)"
COMPARE_OUT_DIR="$OUT_DIR/compare_${SAFE_OPS}_${SAFE_RATIO}_k${K}_M${MAG}"
mkdir -p "$OUT_DIR" "$COMPARE_OUT_DIR"

EPOCH="$("$PYTHON_BIN" - <<PY
import yaml
with open("$CFG", "r") as f:
    cfg = yaml.safe_load(f)
print(cfg.get("epoch", "100"))
PY
)"

MODEL_DIR="$DATA_DIR/$OUTPUT_TIME/model/$EPOCH"
RESULT_DIR="$DATA_DIR/$OUTPUT_TIME/result/$EPOCH"

CFG_BACKUP="$(mktemp /tmp/nav_config_backup.XXXXXX.yaml)"
cp -f "$CFG" "$CFG_BACKUP"

restore_config() {
  cp -f "$CFG_BACKUP" "$CFG"
  rm -f "$CFG_BACKUP"
}
trap restore_config EXIT

check_dataset_exists() {
  local time_id="$1"
  local ds="$DATA_DIR/$time_id/dataset"
  [[ -d "$ds/img" ]] || { echo "[ERR] missing: $ds/img" >&2; exit 1; }
  [[ -f "$ds/vel/data.csv" ]] || { echo "[ERR] missing: $ds/vel/data.csv" >&2; exit 1; }
}

check_split_exists() {
  local time_id="$1"
  local ds="$DATA_DIR/$time_id/dataset"
  [[ -d "$ds/$TRAIN_IMG_DIR" ]] || { echo "[ERR] missing: $ds/$TRAIN_IMG_DIR" >&2; exit 1; }
  [[ -f "$ds/$TRAIN_VEL_DIR/data.csv" ]] || { echo "[ERR] missing: $ds/$TRAIN_VEL_DIR/data.csv" >&2; exit 1; }
  [[ -d "$ds/$TEST_IMG_DIR" ]] || { echo "[ERR] missing: $ds/$TEST_IMG_DIR" >&2; exit 1; }
  [[ -f "$ds/$TEST_VEL_DIR/data.csv" ]] || { echo "[ERR] missing: $ds/$TEST_VEL_DIR/data.csv" >&2; exit 1; }
}

set_cfg_for_aug() {
  local time_id="$1"
  local in_img="$2"
  local in_vel="$3"
  local out_img="$4"
  local out_vel="$5"

  CFG_PATH="$CFG" TIME_ID="$time_id" IN_IMG="$in_img" IN_VEL="$in_vel" OUT_IMG="$out_img" OUT_VEL="$out_vel" "$PYTHON_BIN" - <<'PY'
import os
import yaml

p = os.environ["CFG_PATH"]
with open(p, "r") as f:
    cfg = yaml.safe_load(f)

cfg["time"] = os.environ["TIME_ID"]
cfg["train_times"] = []
cfg["input_aug_dataset_img"] = os.environ["IN_IMG"]
cfg["input_aug_dataset_vel"] = os.environ["IN_VEL"]
cfg["output_aug_dataset_img"] = os.environ["OUT_IMG"]
cfg["output_aug_dataset_vel"] = os.environ["OUT_VEL"]

with open(p, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

run_train_model() {
  local label="$1"
  local load_img="$2"
  local load_vel="$3"
  local save_model="$4"
  echo "[RUN] train $label"
  NAV_TIME="$OUTPUT_TIME" \
  NAV_OUTPUT_TIME="$OUTPUT_TIME" \
  NAV_TRAIN_TIMES="$OUTPUT_TIME" \
  NAV_LOAD_DATASET_IMG="$load_img" \
  NAV_LOAD_DATASET_VEL="$load_vel" \
  NAV_SAVE_MODEL="$save_model" \
  "$PYTHON_BIN" "$LEARN_PY"
}

build_merged_dataset() {
  local out_img="$1"
  local out_vel="$2"
  local mode="$3"
  local aug_img="$4"
  local aug_vel="$5"
  shift 5
  local clean_times=("$@")

  [[ ${#clean_times[@]} -gt 0 ]] || { echo "[ERR] build_merged_dataset requires clean times" >&2; exit 1; }
  local clean_csv
  clean_csv="$(IFS=,; echo "${clean_times[*]}")"

  NAV_PATH="$NAV_DIR" OUTPUT_TIME="$OUTPUT_TIME" CLEAN_TIMES_CSV="$clean_csv" TRAIN_IMG="$TRAIN_IMG_DIR" TRAIN_VEL="$TRAIN_VEL_DIR" OUT_IMG="$out_img" OUT_VEL="$out_vel" AUG_TIME="$CENTER_TIME" AUG_IMG="$aug_img" AUG_VEL="$aug_vel" MODE="$mode" "$PYTHON_BIN" - <<'PY'
import os
import shutil
import pandas as pd

nav = os.environ["NAV_PATH"]
data_root = os.path.join(nav, "data")
output_time = os.environ["OUTPUT_TIME"]
clean_times = [x for x in os.environ["CLEAN_TIMES_CSV"].split(",") if x]
train_img = os.environ["TRAIN_IMG"]
train_vel = os.environ["TRAIN_VEL"]
out_img = os.environ["OUT_IMG"]
out_vel = os.environ["OUT_VEL"]
aug_time = os.environ["AUG_TIME"]
aug_img = os.environ["AUG_IMG"]
aug_vel = os.environ["AUG_VEL"]
mode = os.environ["MODE"]

target_ds = os.path.join(data_root, output_time, "dataset")
out_img_dir = os.path.join(target_ds, out_img)
out_vel_dir = os.path.join(target_ds, out_vel)
out_csv = os.path.join(out_vel_dir, "data.csv")

for p in (out_img_dir, out_vel_dir):
    if os.path.isdir(p):
        shutil.rmtree(p)
    os.makedirs(p, exist_ok=True)

sources = []
for i, time_id in enumerate(clean_times, start=1):
    sources.append((f"clean{i}", time_id, train_img, train_vel))
sources.append((f"{mode}_aug", aug_time, aug_img, aug_vel))

rows = []
missing_rows = 0
for label, time_id, img_dir_name, vel_dir_name in sources:
    ds_root = os.path.join(data_root, time_id, "dataset")
    img_root = os.path.join(ds_root, img_dir_name)
    csv_path = os.path.join(ds_root, vel_dir_name, "data.csv")

    if not os.path.isdir(img_root):
        raise FileNotFoundError(f"img dir not found: {img_root}")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "episode" not in df.columns:
        raise KeyError(f"'episode' column not found: {csv_path}")

    safe_time = time_id.replace(":", "-")
    for _, r in df.iterrows():
        ep = str(r["episode"]).split(".")[0]
        new_ep = f"{label}_{safe_time}_{ep}"

        ok = True
        for view in ("center", "left", "right"):
            src = os.path.join(img_root, f"{ep}_{view}.npy")
            dst = os.path.join(out_img_dir, f"{new_ep}_{view}.npy")
            if not os.path.isfile(src):
                ok = False
                break
            if os.path.lexists(dst):
                os.unlink(dst)
            os.symlink(os.path.abspath(src), dst)

        if not ok:
            missing_rows += 1
            continue

        rec = r.to_dict()
        rec["episode"] = new_ep
        rows.append(rec)

pd.DataFrame(rows).to_csv(out_csv, index=False)
print(f"[DONE] merged rows={len(rows)} missing_rows={missing_rows}")
print(f"[DONE] merged img dir: {out_img_dir}")
print(f"[DONE] merged csv    : {out_csv}")
PY
}

run_taw_aug_selectedops() {
  local out_img="$1"
  local out_vel="$2"
  echo "[RUN] TrivialAugmentWide selected-ops center14_gap"
  echo "      center=$CENTER_TIME K=$K M=$MAG ratio=$MAX_SEVERITY_RATIO ops=$OPS_CSV"
  set_cfg_for_aug "$CENTER_TIME" "$TRAIN_IMG_DIR" "$TRAIN_VEL_DIR" "$out_img" "$out_vel"
  "$PYTHON_BIN" "$TAW_PY" \
    --K "$K" \
    --same3view \
    --num-magnitude-bins "$MAG" \
    --max-severity-ratio "$MAX_SEVERITY_RATIO" \
    --allowed-ops "$OPS_CSV"
}

model_name() {
  echo "model_gpu_trivialaugwide_K${K}_M${MAG}_${SAFE_RATIO}_${SAFE_OPS}_center14_gap_split4000.pt"
}

ZIP_FILES=()
for t in "${ALL_TIMES[@]}"; do
  ZIP_FILES+=("$DATA_DIR/$t.zip")
done

if [[ "$RUN_UNZIP" -eq 1 ]]; then
  echo "[STEP] unzip datasets"
  for z in "${ZIP_FILES[@]}"; do
    [[ -f "$z" ]] || { echo "[ERR] zip not found: $z" >&2; exit 1; }
    unzip -n -q "$z" -d "$DATA_DIR"
  done
else
  echo "[STEP] skip unzip (--skip-unzip)"
fi

for t in "${ALL_TIMES[@]}"; do
  check_dataset_exists "$t"
done

if [[ "$RUN_SPLIT" -eq 1 ]]; then
  echo "[STEP] build train/test split"
  "$PYTHON_BIN" "$SPLIT_PY" \
    --data-root "$DATA_DIR" \
    --times "${ALL_TIMES[@]}" \
    --train-rows "$TRAIN_ROWS" \
    --train-img-dir "$TRAIN_IMG_DIR" \
    --train-vel-dir "$TRAIN_VEL_DIR" \
    --test-img-dir "$TEST_IMG_DIR" \
    --test-vel-dir "$TEST_VEL_DIR" \
    --summary-csv "$OUT_DIR/split_summary.csv"
else
  echo "[STEP] skip split (--skip-split)"
fi

for t in "${ALL_TIMES[@]}"; do
  check_split_exists "$t"
done

{
  echo "scenario,method,k,magnitude_bins,max_severity_ratio,allowed_ops,clean_times"
  echo "center14_gap_selectedops_cap,taw,$K,$MAG,$MAX_SEVERITY_RATIO,$OPS_CSV,$(IFS=,; echo "${CLEAN_TIMES_K3[*]}")"
} > "$OUT_DIR/scenario_train_sources.csv"

AUG_IMG="TrivialAugmentWide_${TRAIN_IMG_DIR}_center14_K${K}_M${MAG}_${SAFE_RATIO}_${SAFE_OPS}"
AUG_VEL="TrivialAugmentWide_${TRAIN_VEL_DIR}_center14_K${K}_M${MAG}_${SAFE_RATIO}_${SAFE_OPS}"
MIX_IMG="mix_taw_center14_gap_K${K}_M${MAG}_${SAFE_RATIO}_${SAFE_OPS}_img"
MIX_VEL="mix_taw_center14_gap_K${K}_M${MAG}_${SAFE_RATIO}_${SAFE_OPS}_vel"
MODEL_NAME="$(model_name)"

if [[ "$RUN_TRAIN" -eq 1 ]]; then
  run_taw_aug_selectedops "$AUG_IMG" "$AUG_VEL"
  build_merged_dataset "$MIX_IMG" "$MIX_VEL" "taw_${SAFE_RATIO}_${SAFE_OPS}" "$AUG_IMG" "$AUG_VEL" "${CLEAN_TIMES_K3[@]}"
  run_train_model "TrivialAugmentWide center14_gap K=${K} M=${MAG} ${SAFE_RATIO} ${OPS_CSV}" "$MIX_IMG" "$MIX_VEL" "$MODEL_NAME"
else
  echo "[STEP] skip augmentation/training (--skip-train)"
fi

MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
[[ -f "$MODEL_PATH" ]] || {
  echo "[ERR] model not found: $MODEL_PATH" >&2
  exit 1
}

if [[ "$RUN_EVAL" -eq 1 ]]; then
  echo "[STEP] evaluate selected-ops model on all 9 test times"
  ALIAS_DIR="$COMPARE_OUT_DIR/model_alias"
  rm -rf "$ALIAS_DIR"
  mkdir -p "$ALIAS_DIR"
  ALIAS_MODEL="$ALIAS_DIR/model_gpu_trivialaugwide_K${K}_${SAFE_RATIO}_${SAFE_OPS}.pt"
  ln -sfn "$MODEL_PATH" "$ALIAS_MODEL"

  {
    echo "model,real_model_path,alias_model_path,k,magnitude_bins,max_severity_ratio,allowed_ops"
    echo "TrivialAugmentWide_K${K},$MODEL_PATH,$ALIAS_MODEL,$K,$MAG,$MAX_SEVERITY_RATIO,$OPS_CSV"
  } > "$COMPARE_OUT_DIR/selectedops_model_map.csv"

  cmp_args=(
    "$COMPARE_PY"
    "--target_time" "$CENTER_TIME"
    "--model_dir" "$ALIAS_DIR"
    "--ks" "$K"
    "--out_dir" "$COMPARE_OUT_DIR"
    "--allow-missing-models"
    "--test-img-dir" "$TEST_IMG_DIR"
    "--test-vel-dir" "$TEST_VEL_DIR"
    "--test_times"
  )
  for t in "${ALL_TIMES[@]}"; do
    cmp_args+=("$t")
  done
  "$PYTHON_BIN" "${cmp_args[@]}"
else
  echo "[STEP] skip evaluation (--skip-eval)"
fi

echo "[DONE] finished selected-ops TAW center14_gap training"
echo "  output_time               : $OUTPUT_TIME"
echo "  allowed_ops               : $OPS_CSV"
echo "  max_severity_ratio        : $MAX_SEVERITY_RATIO"
echo "  model                     : $MODEL_PATH"
echo "  tensorboard result        : $RESULT_DIR"
echo "  scenario map              : $OUT_DIR/scenario_train_sources.csv"
echo "  eval summary_by_model.csv : $COMPARE_OUT_DIR/summary_by_model.csv"
