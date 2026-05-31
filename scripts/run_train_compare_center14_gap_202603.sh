#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$NAV_DIR/data"
CFG="$NAV_DIR/config/config.yaml"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SPLIT_PY="$NAV_DIR/scripts/split_dataset_first_n.py"
LEARN_PY="$NAV_DIR/scripts/learning_surprise.py"
COMPARE_PY="$NAV_DIR/scripts/compare_models_on_test_times.py"
PLOT_ALL_PY="$NAV_DIR/scripts/plot_compare_mae_lines.py"
PLOT_METHOD_PY="$NAV_DIR/scripts/plot_center14_gap_compare.py"
AUGMIX_PY="$NAV_DIR/scripts/edit_dataset/aug_mix.py"
TAW_PY="$NAV_DIR/scripts/edit_dataset/aug_TrivialAugmentWide.py"
RANDAUG_PY="$NAV_DIR/scripts/edit_dataset/rand_augment.py"

TRAIN_ROWS=4000
TRAIN_IMG_DIR="train4000_img"
TRAIN_VEL_DIR="train4000_vel"
TEST_IMG_DIR="test_rest_img"
TEST_VEL_DIR="test_rest_vel"
OUTPUT_TIME="_center14_gap_20260308_0310"
OUT_DIR=""
EPOCH_OVERRIDE="${NAV_EPOCH:-}"
RUN_UNZIP=1
RUN_SPLIT=1
RUN_TRAIN=1
RUN_PLOT=1
RAND_N=2
RAND_M=9
RAND_P=0.5
CENTER_TIME="20260308_140206"
KS=(3 5 7 9)

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

usage() {
  cat <<'EOF'
Usage:
  run_train_compare_center14_gap_202603.sh [options]

Options:
  --train-rows N            default: 4000
  --output-time NAME        default: _center14_gap_20260308_0310
  --out-dir DIR             default: data/<output-time>
  --data-dir DIR            default: env NAV_DATA_DIR or nav_cloning/data
  --epoch N                 default: env NAV_EPOCH or config epoch
  --skip-unzip              do not extract zip files
  --skip-split              do not rebuild train/test split dirs
  --skip-train              do not retrain models
  --skip-plot               export CSV only
  --rand-n N                RandAugment N (default: 2)
  --rand-m N                RandAugment M (default: 9)
  --rand-p P                RandAugment p (default: 0.5)
  -h, --help

Experiment:
  baseline:
    - train on all 9 clean train splits
  K=3:
    - remove 13:00 and 15:00 from clean train splits
    - add 14:00 augmentation with K=3
  K=5:
    - remove 12:00, 13:00, 15:00, 16:00
    - add 14:00 augmentation with K=5
  K=7:
    - remove 11:00, 12:00, 13:00, 15:00, 16:00, 17:00
    - add 14:00 augmentation with K=7
  K=9:
    - keep only 14:00 clean split
    - add 14:00 augmentation with K=9

For each augmentation method, this script trains:
  baseline + K=3/5/7/9
Then it evaluates on the non-augmented test splits of all 9 times.
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
    --data-dir)
      DATA_DIR="${2:?missing value for --data-dir}"
      shift 2
      ;;
    --epoch)
      EPOCH_OVERRIDE="${2:?missing value for --epoch}"
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
    --rand-n)
      RAND_N="${2:?missing value for --rand-n}"
      shift 2
      ;;
    --rand-m)
      RAND_M="${2:?missing value for --rand-m}"
      shift 2
      ;;
    --rand-p)
      RAND_P="${2:?missing value for --rand-p}"
      shift 2
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

DATA_DIR="${NAV_DATA_DIR:-$DATA_DIR}"

for f in "$CFG" "$SPLIT_PY" "$LEARN_PY" "$COMPARE_PY" "$PLOT_ALL_PY" "$PLOT_METHOD_PY" "$AUGMIX_PY" "$TAW_PY" "$RANDAUG_PY"; do
  [[ -f "$f" ]] || { echo "[ERR] missing file: $f" >&2; exit 1; }
done

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$DATA_DIR/$OUTPUT_TIME"
fi
COMPARE_OUT_DIR="$OUT_DIR/compare"
mkdir -p "$OUT_DIR" "$COMPARE_OUT_DIR"

if [[ -n "$EPOCH_OVERRIDE" ]]; then
  EPOCH="$EPOCH_OVERRIDE"
else
  EPOCH="$("$PYTHON_BIN" - <<PY
import yaml
with open("$CFG", "r") as f:
    cfg = yaml.safe_load(f)
print(cfg.get("epoch", "100"))
PY
)"
fi
[[ "$EPOCH" =~ ^[0-9]+$ ]] || { echo "[ERR] --epoch must be integer: $EPOCH" >&2; exit 1; }
export NAV_EPOCH="$EPOCH"
export NAV_DATA_DIR="$DATA_DIR"

MODEL_DIR="$DATA_DIR/$OUTPUT_TIME/model/$EPOCH"
RESULT_DIR="$DATA_DIR/$OUTPUT_TIME/result/$EPOCH"
TIMES_CSV="$(IFS=,; echo "${ALL_TIMES[*]}")"

CFG_BACKUP="$(mktemp /tmp/nav_config_backup.XXXXXX.yaml)"
cp -f "$CFG" "$CFG_BACKUP"

restore_config() {
  cp -f "$CFG_BACKUP" "$CFG"
  rm -f "$CFG_BACKUP"
}
trap restore_config EXIT

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

run_center_aug() {
  local name="$1"
  local py="$2"
  local out_img="$3"
  local out_vel="$4"
  local k="$5"
  shift 5
  echo "[RUN] $name center=$CENTER_TIME K=$k"
  set_cfg_for_aug "$CENTER_TIME" "$TRAIN_IMG_DIR" "$TRAIN_VEL_DIR" "$out_img" "$out_vel"
  "$PYTHON_BIN" "$py" --K "$k" "$@"
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

contains_time() {
  local needle="$1"
  shift
  local x
  for x in "$@"; do
    [[ "$x" == "$needle" ]] && return 0
  done
  return 1
}

clean_times_for_k() {
  local k="$1"
  local excluded=()
  case "$k" in
    3)
      excluded=("20260308_130242" "20260308_150431")
      ;;
    5)
      excluded=("20260308_120428" "20260308_130242" "20260308_150431" "20260308_160405")
      ;;
    7)
      excluded=("20260308_111130" "20260308_120428" "20260308_130242" "20260308_150431" "20260308_160405" "20260310_170214")
      ;;
    9)
      excluded=("20260308_100554" "20260308_111130" "20260308_120428" "20260308_130242" "20260308_150431" "20260308_160405" "20260310_170214" "20260310_180327")
      ;;
    *)
      echo "[ERR] unsupported K for center-gap scenario: $k" >&2
      exit 1
      ;;
  esac

  local kept=()
  local t
  for t in "${ALL_TIMES[@]}"; do
    if ! contains_time "$t" "${excluded[@]}"; then
      kept+=("$t")
    fi
  done
  printf "%s\n" "${kept[@]}"
}

build_merged_dataset() {
  local out_img="$1"
  local out_vel="$2"
  local mode="$3"
  local aug_img="${4:-}"
  local aug_vel="${5:-}"
  shift 5 || true
  local clean_times=("$@")

  if [[ ${#clean_times[@]} -eq 0 ]]; then
    echo "[ERR] build_merged_dataset requires at least one clean time" >&2
    exit 1
  fi

  local clean_csv aug_enabled=0
  clean_csv="$(IFS=,; echo "${clean_times[*]}")"
  if [[ -n "$aug_img" && -n "$aug_vel" ]]; then
    aug_enabled=1
  fi

  DATA_ROOT="$DATA_DIR" OUTPUT_TIME="$OUTPUT_TIME" CLEAN_TIMES_CSV="$clean_csv" TRAIN_IMG="$TRAIN_IMG_DIR" TRAIN_VEL="$TRAIN_VEL_DIR" OUT_IMG="$out_img" OUT_VEL="$out_vel" AUG_ENABLED="$aug_enabled" AUG_TIME="$CENTER_TIME" AUG_IMG="$aug_img" AUG_VEL="$aug_vel" MODE="$mode" "$PYTHON_BIN" - <<'PY'
import os
import shutil
import pandas as pd

data_root = os.environ["DATA_ROOT"]
output_time = os.environ["OUTPUT_TIME"]
clean_times = [x for x in os.environ["CLEAN_TIMES_CSV"].split(",") if x]
train_img = os.environ["TRAIN_IMG"]
train_vel = os.environ["TRAIN_VEL"]
out_img = os.environ["OUT_IMG"]
out_vel = os.environ["OUT_VEL"]
aug_enabled = os.environ["AUG_ENABLED"] == "1"
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
if aug_enabled:
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
  echo "scenario,method,k,clean_times"
  echo "baseline,baseline,0,$TIMES_CSV"
} > "$OUT_DIR/scenario_train_sources.csv"

if [[ "$RUN_TRAIN" -eq 1 ]]; then
  echo "[STEP] build baseline merged dataset"
  build_merged_dataset \
    "mix_baseline_center14_allclean9_img" \
    "mix_baseline_center14_allclean9_vel" \
    "baseline" \
    "" \
    "" \
    "${ALL_TIMES[@]}"

  run_train_model \
    "baseline(center14 reference, all clean times)" \
    "mix_baseline_center14_allclean9_img" \
    "mix_baseline_center14_allclean9_vel" \
    "model_gpu_baseline_center14_allclean9_split4000.pt"

  for k in "${KS[@]}"; do
    mapfile -t CLEAN_TIMES_K < <(clean_times_for_k "$k")
    echo "[INFO] K=$k clean times: $(IFS=,; echo "${CLEAN_TIMES_K[*]}")"

    echo "center14_gap,all,$k,$(IFS=,; echo "${CLEAN_TIMES_K[*]}")" >> "$OUT_DIR/scenario_train_sources.csv"

    augmix_img="augmix_${TRAIN_IMG_DIR}_center14_K${k}"
    augmix_vel="augmix_${TRAIN_VEL_DIR}_center14_K${k}"
    taw_img="TrivialAugmentWide_${TRAIN_IMG_DIR}_center14_K${k}"
    taw_vel="TrivialAugmentWide_${TRAIN_VEL_DIR}_center14_K${k}"
    rand_img="randaugment_${TRAIN_IMG_DIR}_center14_K${k}"
    rand_vel="randaugment_${TRAIN_VEL_DIR}_center14_K${k}"

    run_center_aug "AugMix" "$AUGMIX_PY" "$augmix_img" "$augmix_vel" "$k"
    run_center_aug "TrivialAugmentWide" "$TAW_PY" "$taw_img" "$taw_vel" "$k" --same3view
    run_center_aug "RandAugment" "$RANDAUG_PY" "$rand_img" "$rand_vel" "$k" --N "$RAND_N" --M "$RAND_M" --p "$RAND_P" --sync_views

    build_merged_dataset \
      "mix_augmix_center14_gap_K${k}_img" \
      "mix_augmix_center14_gap_K${k}_vel" \
      "augmix" \
      "$augmix_img" \
      "$augmix_vel" \
      "${CLEAN_TIMES_K[@]}"
    build_merged_dataset \
      "mix_taw_center14_gap_K${k}_img" \
      "mix_taw_center14_gap_K${k}_vel" \
      "taw" \
      "$taw_img" \
      "$taw_vel" \
      "${CLEAN_TIMES_K[@]}"
    build_merged_dataset \
      "mix_randaugment_center14_gap_K${k}_img" \
      "mix_randaugment_center14_gap_K${k}_vel" \
      "rand" \
      "$rand_img" \
      "$rand_vel" \
      "${CLEAN_TIMES_K[@]}"

    run_train_model \
      "augmix(center14 gap K=$k)" \
      "mix_augmix_center14_gap_K${k}_img" \
      "mix_augmix_center14_gap_K${k}_vel" \
      "model_gpu_augmix_K${k}_center14_gap_split4000.pt"

    run_train_model \
      "TrivialAugmentWide(center14 gap K=$k)" \
      "mix_taw_center14_gap_K${k}_img" \
      "mix_taw_center14_gap_K${k}_vel" \
      "model_gpu_trivialaugwide_K${k}_center14_gap_split4000.pt"

    run_train_model \
      "rand_augment(center14 gap K=$k)" \
      "mix_randaugment_center14_gap_K${k}_img" \
      "mix_randaugment_center14_gap_K${k}_vel" \
      "model_gpu_randaugment_K${k}_center14_gap_split4000.pt"
  done
else
  echo "[STEP] skip train (--skip-train)"
fi

[[ -d "$MODEL_DIR" ]] || { echo "[ERR] model dir not found: $MODEL_DIR" >&2; exit 1; }

echo "[STEP] evaluate models on test split"
"$PYTHON_BIN" "$COMPARE_PY" \
  --model_dir "$MODEL_DIR" \
  --ks "3,5,7,9" \
  --out_dir "$COMPARE_OUT_DIR" \
  --test-img-dir "$TEST_IMG_DIR" \
  --test-vel-dir "$TEST_VEL_DIR" \
  --test_times "${ALL_TIMES[@]}"

if [[ "$RUN_PLOT" -eq 1 ]]; then
  echo "[STEP] plot all-model overview"
  "$PYTHON_BIN" "$PLOT_ALL_PY" \
    --summary_csv "$COMPARE_OUT_DIR/summary_by_time_model.csv" \
    --out_dir "$COMPARE_OUT_DIR"

  echo "[STEP] plot per-method center14 gap comparison"
  "$PYTHON_BIN" "$PLOT_METHOD_PY" \
    --summary_csv "$COMPARE_OUT_DIR/summary_by_time_model.csv" \
    --out_dir "$COMPARE_OUT_DIR"
else
  echo "[STEP] skip plot (--skip-plot)"
fi

echo "[DONE] output_time        : $OUTPUT_TIME"
echo "[DONE] model_dir           : $MODEL_DIR"
echo "[DONE] tensorboard result  : $RESULT_DIR"
echo "[DONE] split_summary.csv   : $OUT_DIR/split_summary.csv"
echo "[DONE] scenario map        : $OUT_DIR/scenario_train_sources.csv"
echo "[DONE] summary_by_time     : $COMPARE_OUT_DIR/summary_by_time_model.csv"
echo "[DONE] summary_by_model    : $COMPARE_OUT_DIR/summary_by_model.csv"
echo "[DONE] detail_per_sample   : $COMPARE_OUT_DIR/detail_per_sample.csv"
