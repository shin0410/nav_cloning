#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CFG="$NAV_DIR/config/config.yaml"

TAW_PY="$NAV_DIR/scripts/edit_dataset/aug_TrivialAugmentWide.py"
LEARN_PY="$NAV_DIR/scripts/learning_surprise.py"
COMPARE_PY="$NAV_DIR/scripts/compare_models_on_test_times.py"
PLOT_OP_PY="$NAV_DIR/scripts/plot_taw_op_compare.py"

PYTHON_BIN="${PYTHON_BIN:-python3}"

TARGET_TIME="20260112_14:08:03"
K=5
MAG=31
OPS_CSV="AutoContrast,Brightness,Color,Contrast,Equalize,Posterize,Sharpness,Solarize"
RUN_TRAIN=1
OUT_DIR=""

CLEAN_TIMES=(
  "20260112_10:01:45"
  "20260112_11:02:15"
  "20260112_17:11:48"
  "20260112_18:10:41"
)

TEST_TIMES=(
  "20260112_10:05:46"
  "20260112_11:09:30"
  "20260112_12:06:28"
  "20260112_13:09:38"
  "20260112_14:11:47"
  "20260112_15:07:57"
  "20260112_16:19:24"
  "20260112_17:15:12"
  "20260112_18:14:00"
)

usage() {
  cat <<'EOF'
Usage:
  run_taw_singleop_k5_compare.sh [options]

Options:
  --target-time TIME         default: 20260112_14:08:03
  --ops LIST                 default: AutoContrast,Brightness,Color,Contrast,Equalize,Posterize,Sharpness,Solarize
  --out-dir DIR              output root (default: auto under data/)
  --skip-train               skip training and run compare/plot only
  -h, --help

Fixed settings:
  K=5
  TrivialAugmentWide num-magnitude-bins=31
  each run uses only one operation from --ops

Flow:
  1) Augment target dataset with single op (K=5, M=31)
  2) Merge with clean4 datasets
  3) Train one model per operation
  4) Evaluate all operation models on test times
  5) Export MAE tables and line plot
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-time)
      TARGET_TIME="${2:?missing value for --target-time}"
      shift 2
      ;;
    --ops)
      OPS_CSV="${2:?missing value for --ops}"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="${2:?missing value for --out-dir}"
      shift 2
      ;;
    --skip-train)
      RUN_TRAIN=0
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

for f in "$CFG" "$TAW_PY" "$LEARN_PY" "$COMPARE_PY" "$PLOT_OP_PY"; do
  [[ -f "$f" ]] || { echo "[ERR] missing file: $f" >&2; exit 1; }
done

check_dataset_exists() {
  local time_id="$1"
  local ds="$NAV_DIR/data/$time_id/dataset"
  [[ -d "$ds/img" ]] || { echo "[ERR] missing: $ds/img" >&2; exit 1; }
  [[ -f "$ds/vel/data.csv" ]] || { echo "[ERR] missing: $ds/vel/data.csv" >&2; exit 1; }
}

check_dataset_exists "$TARGET_TIME"
for t in "${CLEAN_TIMES[@]}"; do
  check_dataset_exists "$t"
done

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

EPOCH="$("$PYTHON_BIN" - <<PY
import yaml
with open("$CFG", "r") as f:
    cfg = yaml.safe_load(f)
print(cfg.get("epoch", "100"))
PY
)"

CFG_BACKUP="$(mktemp /tmp/nav_config_backup.XXXXXX.yaml)"
cp -f "$CFG" "$CFG_BACKUP"

restore_config() {
  cp -f "$CFG_BACKUP" "$CFG"
  rm -f "$CFG_BACKUP"
  echo "[CFG] restored: $CFG"
}
trap restore_config EXIT

set_cfg_for_aug() {
  local in_img="$1"
  local in_vel="$2"
  local out_img="$3"
  local out_vel="$4"

  CFG_PATH="$CFG" TIME_ID="$TARGET_TIME" IN_IMG="$in_img" IN_VEL="$in_vel" OUT_IMG="$out_img" OUT_VEL="$out_vel" "$PYTHON_BIN" - <<'PY'
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

set_cfg_for_train() {
  local load_img="$1"
  local load_vel="$2"
  local save_model="$3"

  CFG_PATH="$CFG" TIME_ID="$TARGET_TIME" LOAD_IMG="$load_img" LOAD_VEL="$load_vel" SAVE_MODEL="$save_model" "$PYTHON_BIN" - <<'PY'
import os
import yaml

p = os.environ["CFG_PATH"]
with open(p, "r") as f:
    cfg = yaml.safe_load(f)

cfg["time"] = os.environ["TIME_ID"]
cfg["train_times"] = []
cfg["load_dataset_img"] = os.environ["LOAD_IMG"]
cfg["load_dataset_vel"] = os.environ["LOAD_VEL"]
cfg["save_model"] = os.environ["SAVE_MODEL"]
cfg["load_model"] = os.environ["SAVE_MODEL"]

with open(p, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

build_mixed_dataset() {
  local tag="$1"
  local aug_img="$2"
  local aug_vel="$3"
  local out_img="$4"
  local out_vel="$5"
  shift 5
  local clean_times=("$@")

  [[ ${#clean_times[@]} -gt 0 ]] || { echo "[ERR] build_mixed_dataset requires clean times" >&2; exit 1; }
  local clean_csv
  clean_csv="$(IFS=,; echo "${clean_times[*]}")"
  echo "[RUN] build mixed dataset ($tag): target aug + clean times = $clean_csv"

  NAV_PATH="$NAV_DIR" TARGET_TIME="$TARGET_TIME" CLEAN_TIMES_CSV="$clean_csv" AUG_IMG="$aug_img" AUG_VEL="$aug_vel" OUT_IMG="$out_img" OUT_VEL="$out_vel" "$PYTHON_BIN" - <<'PY'
import os
import shutil
import pandas as pd

nav = os.environ["NAV_PATH"]
data_root = os.path.join(nav, "data")
target = os.environ["TARGET_TIME"]
clean_times = [x for x in os.environ["CLEAN_TIMES_CSV"].split(",") if x]
aug_img = os.environ["AUG_IMG"]
aug_vel = os.environ["AUG_VEL"]
out_img = os.environ["OUT_IMG"]
out_vel = os.environ["OUT_VEL"]

target_ds = os.path.join(data_root, target, "dataset")
out_img_dir = os.path.join(target_ds, out_img)
out_vel_dir = os.path.join(target_ds, out_vel)
out_csv = os.path.join(out_vel_dir, "data.csv")

for p in (out_img_dir, out_vel_dir):
    if os.path.isdir(p):
        shutil.rmtree(p)
    os.makedirs(p, exist_ok=True)

sources = [("aug", target, aug_img, aug_vel)]
for i, t in enumerate(clean_times, start=1):
    sources.append((f"clean{i}", t, "img", "vel"))

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

op_safe_name() {
  local op="$1"
  echo "$op" | tr -c 'A-Za-z0-9_' '_'
}

run_taw_aug() {
  local out_img="$1"
  local out_vel="$2"
  local op="$3"
  echo "[RUN] TrivialAugmentWide single-op augmentation (time=$TARGET_TIME, K=$K, M=$MAG, op=$op)"
  set_cfg_for_aug "img" "vel" "$out_img" "$out_vel"
  "$PYTHON_BIN" "$TAW_PY" \
    --K "$K" \
    --same3view \
    --num-magnitude-bins "$MAG" \
    --allowed-ops "$op"
}

run_train() {
  local name="$1"
  local load_img="$2"
  local load_vel="$3"
  local model_name="$4"
  echo "[RUN] train $name (img=$load_img vel=$load_vel save=$model_name)"
  set_cfg_for_train "$load_img" "$load_vel" "$model_name"
  "$PYTHON_BIN" "$LEARN_PY"
}

SAFE_TARGET="${TARGET_TIME//:/-}"
SAFE_C1="${CLEAN_TIMES[0]//:/-}"
SAFE_C2="${CLEAN_TIMES[1]//:/-}"
SAFE_C3="${CLEAN_TIMES[2]//:/-}"
SAFE_C4="${CLEAN_TIMES[3]//:/-}"
MODEL_DIR="$NAV_DIR/data/$TARGET_TIME/model/$EPOCH"

model_name_for_op() {
  local op="$1"
  local safe_op
  safe_op="$(op_safe_name "$op")"
  echo "model_gpu_trivialaugwide_K${K}_M${MAG}_OP${safe_op}_plusclean4_${SAFE_TARGET}_${SAFE_C1}_${SAFE_C2}_${SAFE_C3}_${SAFE_C4}.pt"
}

if [[ "$RUN_TRAIN" -eq 1 ]]; then
  for op in "${OPS[@]}"; do
    safe_op="$(op_safe_name "$op")"
    aug_img="TrivialAugmentWide_img_K${K}_M${MAG}_OP${safe_op}"
    aug_vel="TrivialAugmentWide_vel_K${K}_M${MAG}_OP${safe_op}"
    mix_img="mix_taw_img_K${K}_M${MAG}_OP${safe_op}_plus4clean"
    mix_vel="mix_taw_vel_K${K}_M${MAG}_OP${safe_op}_plus4clean"
    model_name="$(model_name_for_op "$op")"

    run_taw_aug "$aug_img" "$aug_vel" "$op"
    build_mixed_dataset "TrivialAugmentWide_OP${safe_op}" "$aug_img" "$aug_vel" "$mix_img" "$mix_vel" "${CLEAN_TIMES[@]}"
    run_train "TrivialAugmentWide(K${K},M${MAG},OP=${op}+clean4)" "$mix_img" "$mix_vel" "$model_name"
  done
else
  echo "[STEP] skip training (--skip-train)"
fi

for op in "${OPS[@]}"; do
  model_path="$MODEL_DIR/$(model_name_for_op "$op")"
  [[ -f "$model_path" ]] || {
    echo "[ERR] model not found for op=$op: $model_path" >&2
    exit 1
  }
done

if [[ -z "$OUT_DIR" ]]; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  OUT_DIR="$NAV_DIR/data/_taw_singleop_k5_compare_${SAFE_TARGET}_${STAMP}"
fi
mkdir -p "$OUT_DIR"

ALIAS_DIR="$OUT_DIR/model_alias"
rm -rf "$ALIAS_DIR"
mkdir -p "$ALIAS_DIR"

MAP_CSV="$OUT_DIR/taw_singleop_model_map.csv"
echo "k_alias,op_name,k,magnitude_bins,real_model_path,alias_model_path" > "$MAP_CSV"

K_ALIAS_LIST=()
k_alias=101
for op in "${OPS[@]}"; do
  safe_op="$(op_safe_name "$op")"
  real_model="$MODEL_DIR/$(model_name_for_op "$op")"
  alias_model="$ALIAS_DIR/model_gpu_trivialaugwide_K${k_alias}_${safe_op}.pt"
  ln -sfn "$real_model" "$alias_model"
  echo "${k_alias},${op},${K},${MAG},${real_model},${alias_model}" >> "$MAP_CSV"
  K_ALIAS_LIST+=("$k_alias")
  k_alias=$((k_alias + 1))
done

KS_FOR_COMPARE="$(IFS=,; echo "${K_ALIAS_LIST[*]}")"

echo "[STEP] evaluating TrivialAugmentWide single-op sweep"
cmp_args=(
  "$COMPARE_PY"
  "--target_time" "$TARGET_TIME"
  "--model_dir" "$ALIAS_DIR"
  "--ks" "$KS_FOR_COMPARE"
  "--out_dir" "$OUT_DIR"
  "--allow-missing-models"
  "--test_times"
)
for t in "${TEST_TIMES[@]}"; do
  cmp_args+=("$t")
done
"$PYTHON_BIN" "${cmp_args[@]}"

echo "[STEP] plotting TrivialAugmentWide single-op sweep"
"$PYTHON_BIN" "$PLOT_OP_PY" \
  --summary_csv "$OUT_DIR/summary_by_time_model.csv" \
  --op_map_csv "$MAP_CSV" \
  --out_dir "$OUT_DIR"

echo "[DONE] finished TrivialAugmentWide single-op sweep (K=$K, M=$MAG)"
echo "  model map                     : $MAP_CSV"
echo "  summary_by_time_model.csv     : $OUT_DIR/summary_by_time_model.csv"
echo "  summary_by_model.csv          : $OUT_DIR/summary_by_model.csv"
echo "  mae_table_taw_op_compare.csv  : $OUT_DIR/mae_table_taw_op_compare.csv"
echo "  summary_taw_op_overall.csv    : $OUT_DIR/summary_taw_op_overall.csv"
echo "  plot                          : $OUT_DIR/mae_taw_op_compare.png"

