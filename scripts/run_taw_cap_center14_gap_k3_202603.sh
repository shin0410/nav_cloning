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
PLOT_CAP_PY="$NAV_DIR/scripts/plot_taw_cap_compare.py"

PYTHON_BIN="${PYTHON_BIN:-python3}"

TRAIN_ROWS=4000
TRAIN_IMG_DIR="train4000_img"
TRAIN_VEL_DIR="train4000_vel"
TEST_IMG_DIR="test_rest_img"
TEST_VEL_DIR="test_rest_vel"
OUTPUT_TIME="_center14_gap_taw_cap_k3_20260308_0310"
OUT_DIR=""
RUN_UNZIP=1
RUN_SPLIT=1
RUN_TRAIN=1
RUN_PLOT=1

CENTER_TIME="20260308_140206"
K=3
MAG=31
CAP_RATIOS_CSV="1.0,0.75,0.5,0.25,0.1"

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
  run_taw_cap_center14_gap_k3_202603.sh [options]

Options:
  --train-rows N              default: 4000
  --output-time NAME          default: _center14_gap_taw_cap_k3_20260308_0310
  --out-dir DIR               default: data/<output-time>
  --mag N                     TrivialAugmentWide num-magnitude-bins (default: 31)
  --max-severity-ratios LIST  default: 1.0,0.75,0.5,0.25,0.1
  --skip-unzip                do not extract zip files
  --skip-split                do not rebuild train/test split dirs
  --skip-train                skip training and run compare/plot only
  --skip-plot                 export CSV only
  -h, --help

Scenario:
  center14_gap + TrivialAugmentWide max-severity cap sweep
  fixed K=3, center time = 20260308_140206
  clean train times = all except 13:00 and 15:00

Meaning of --max-severity-ratios:
  1.0  -> full original severity range
  0.5  -> only lower half of severity levels are sampled
  0.25 -> only lower quarter of severity levels are sampled
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
    --max-severity-ratios)
      CAP_RATIOS_CSV="${2:?missing value for --max-severity-ratios}"
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
    --skip-plot)
      RUN_PLOT=0
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

for f in "$CFG" "$SPLIT_PY" "$TAW_PY" "$LEARN_PY" "$COMPARE_PY" "$PLOT_CAP_PY"; do
  [[ -f "$f" ]] || { echo "[ERR] missing file: $f" >&2; exit 1; }
done

[[ "$MAG" =~ ^[0-9]+$ ]] || { echo "[ERR] --mag must be integer: $MAG" >&2; exit 1; }
(( MAG >= 2 )) || { echo "[ERR] --mag must be >= 2" >&2; exit 1; }

parse_cap_ratios() {
  local csv="$1"
  mapfile -t CAP_RATIOS < <(CAP_RATIOS_CSV="$csv" "$PYTHON_BIN" - <<'PY'
import os

tokens = os.environ["CAP_RATIOS_CSV"].split(",")
seen = set()
out = []
for tok in tokens:
    s = tok.strip()
    if not s:
        continue
    try:
        v = float(s)
    except Exception:
        raise SystemExit(f"[ERR] invalid ratio token: {tok}")
    if not (0.0 <= v <= 1.0):
        raise SystemExit(f"[ERR] ratio must be in [0.0, 1.0]: {tok}")
    key = round(v, 12)
    if key in seen:
        continue
    seen.add(key)
    out.append(v)

if not out:
    raise SystemExit("[ERR] no valid --max-severity-ratios values")

for v in out:
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    print(s if s else "0")
PY
)
}

ratio_safe_name() {
  local ratio="$1"
  RATIO="$ratio" "$PYTHON_BIN" - <<'PY'
import os
ratio = float(os.environ["RATIO"])
print(f"R{int(round(ratio * 1000)):04d}")
PY
}

ratio_label() {
  local ratio="$1"
  local mag="$2"
  RATIO="$ratio" MAG="$mag" "$PYTHON_BIN" - <<'PY'
import math
import os

ratio = float(os.environ["RATIO"])
mag = int(os.environ["MAG"])
max_level = max(0, min(mag - 1, int(math.floor((mag - 1) * ratio))))
print(f"r={ratio:.2f} (<=bin {max_level})")
PY
}

ratio_max_level() {
  local ratio="$1"
  local mag="$2"
  RATIO="$ratio" MAG="$mag" "$PYTHON_BIN" - <<'PY'
import math
import os
ratio = float(os.environ["RATIO"])
mag = int(os.environ["MAG"])
print(max(0, min(mag - 1, int(math.floor((mag - 1) * ratio)))))
PY
}

parse_cap_ratios "$CAP_RATIOS_CSV"

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$DATA_DIR/$OUTPUT_TIME"
fi
COMPARE_OUT_DIR="$OUT_DIR/compare_taw_cap_k3_M${MAG}"
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

run_taw_aug_cap() {
  local out_img="$1"
  local out_vel="$2"
  local cap_ratio="$3"
  echo "[RUN] TrivialAugmentWide cap-sweep center14_gap (center=$CENTER_TIME K=$K M=$MAG max_ratio=$cap_ratio)"
  set_cfg_for_aug "$CENTER_TIME" "$TRAIN_IMG_DIR" "$TRAIN_VEL_DIR" "$out_img" "$out_vel"
  "$PYTHON_BIN" "$TAW_PY" \
    --K "$K" \
    --same3view \
    --num-magnitude-bins "$MAG" \
    --max-severity-ratio "$cap_ratio"
}

model_name_for_ratio() {
  local ratio="$1"
  local safe_ratio
  safe_ratio="$(ratio_safe_name "$ratio")"
  echo "model_gpu_trivialaugwide_K${K}_M${MAG}_${safe_ratio}_center14_gap_split4000.pt"
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
  echo "scenario,method,k,magnitude_bins,max_severity_ratio,max_severity_level,clean_times"
  for ratio in "${CAP_RATIOS[@]}"; do
    echo "center14_gap_cap,taw,$K,$MAG,$ratio,$(ratio_max_level "$ratio" "$MAG"),$(IFS=,; echo "${CLEAN_TIMES_K3[*]}")"
  done
} > "$OUT_DIR/scenario_train_sources.csv"

if [[ "$RUN_TRAIN" -eq 1 ]]; then
  for ratio in "${CAP_RATIOS[@]}"; do
    safe_ratio="$(ratio_safe_name "$ratio")"
    aug_img="TrivialAugmentWide_${TRAIN_IMG_DIR}_center14_K${K}_M${MAG}_${safe_ratio}"
    aug_vel="TrivialAugmentWide_${TRAIN_VEL_DIR}_center14_K${K}_M${MAG}_${safe_ratio}"
    mix_img="mix_taw_center14_gap_K${K}_M${MAG}_${safe_ratio}_img"
    mix_vel="mix_taw_center14_gap_K${K}_M${MAG}_${safe_ratio}_vel"
    model_name="$(model_name_for_ratio "$ratio")"

    run_taw_aug_cap "$aug_img" "$aug_vel" "$ratio"
    build_merged_dataset "$mix_img" "$mix_vel" "taw_cap_${safe_ratio}" "$aug_img" "$aug_vel" "${CLEAN_TIMES_K3[@]}"
    run_train_model "TrivialAugmentWide center14_gap K=${K} M=${MAG} cap_ratio=${ratio}" "$mix_img" "$mix_vel" "$model_name"
  done
else
  echo "[STEP] skip training (--skip-train)"
fi

for ratio in "${CAP_RATIOS[@]}"; do
  model_path="$MODEL_DIR/$(model_name_for_ratio "$ratio")"
  [[ -f "$model_path" ]] || {
    echo "[ERR] model not found for cap_ratio=$ratio: $model_path" >&2
    exit 1
  }
done

ALIAS_DIR="$COMPARE_OUT_DIR/model_alias"
rm -rf "$ALIAS_DIR"
mkdir -p "$ALIAS_DIR"

MAP_CSV="$COMPARE_OUT_DIR/taw_cap_model_map.csv"
echo "k_alias,cap_ratio,max_severity_level,label,real_model_path,alias_model_path" > "$MAP_CSV"

K_ALIAS_LIST=()
k_alias=201
for ratio in "${CAP_RATIOS[@]}"; do
  real_model="$MODEL_DIR/$(model_name_for_ratio "$ratio")"
  alias_model="$ALIAS_DIR/model_gpu_trivialaugwide_K${k_alias}_capscan.pt"
  label="$(ratio_label "$ratio" "$MAG")"
  max_level="$(ratio_max_level "$ratio" "$MAG")"
  ln -sfn "$real_model" "$alias_model"
  echo "${k_alias},${ratio},${max_level},${label},${real_model},${alias_model}" >> "$MAP_CSV"
  K_ALIAS_LIST+=("$k_alias")
  k_alias=$((k_alias + 1))
done

KS_FOR_COMPARE="$(IFS=,; echo "${K_ALIAS_LIST[*]}")"

echo "[STEP] evaluating center14_gap TAW max-severity cap sweep"
cmp_args=(
  "$COMPARE_PY"
  "--target_time" "$CENTER_TIME"
  "--model_dir" "$ALIAS_DIR"
  "--ks" "$KS_FOR_COMPARE"
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

if [[ "$RUN_PLOT" -eq 1 ]]; then
  echo "[STEP] plotting center14_gap TAW max-severity cap sweep"
  "$PYTHON_BIN" "$PLOT_CAP_PY" \
    --summary_csv "$COMPARE_OUT_DIR/summary_by_time_model.csv" \
    --cap_map_csv "$MAP_CSV" \
    --title "MAE over test datasets (center14 gap, TAW max-severity cap, K=${K}, M=${MAG})" \
    --out_dir "$COMPARE_OUT_DIR"
else
  echo "[STEP] skip plot (--skip-plot)"
fi

echo "[DONE] finished center14_gap TAW max-severity cap sweep (K=$K, M=$MAG)"
echo "  output_time                 : $OUTPUT_TIME"
echo "  model_dir                   : $MODEL_DIR"
echo "  tensorboard result          : $RESULT_DIR"
echo "  split_summary.csv           : $OUT_DIR/split_summary.csv"
echo "  scenario map                : $OUT_DIR/scenario_train_sources.csv"
echo "  model map                   : $MAP_CSV"
echo "  summary_by_time_model.csv   : $COMPARE_OUT_DIR/summary_by_time_model.csv"
echo "  summary_by_model.csv        : $COMPARE_OUT_DIR/summary_by_model.csv"
echo "  mae_table_taw_cap_compare.csv : $COMPARE_OUT_DIR/mae_table_taw_cap_compare.csv"
echo "  summary_taw_cap_overall.csv : $COMPARE_OUT_DIR/summary_taw_cap_overall.csv"
echo "  plot                        : $COMPARE_OUT_DIR/mae_taw_cap_compare.png"
