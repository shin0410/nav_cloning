#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CFG="$NAV_DIR/config/config.yaml"

AUGMIX_PY="$NAV_DIR/scripts/edit_dataset/aug_mix.py"
TAW_PY="$NAV_DIR/scripts/edit_dataset/aug_TrivialAugmentWide.py"
RANDAUG_PY="$NAV_DIR/scripts/edit_dataset/rand_augment.py"
LEARN_PY="$NAV_DIR/scripts/learning_surprise.py"

PYTHON_BIN="${PYTHON_BIN:-python3}"

TARGET_TIME="20260112_14:08:03"

# base4 scenario (existing pattern): baseline + 3 aug models
K=9
RAND_N=2
RAND_M=9
RAND_P=0.5
BASELINE_CLEAN_TIMES=(
  "20260112_10:01:45"
  "20260112_11:02:15"
  "20260112_12:03:40"
  "20260112_13:06:37"
  "20260112_14:08:03"
  "20260112_15:03:18"
  "20260112_16:15:51"
  "20260112_17:11:48"
  "20260112_18:10:41"
)

# k7clean2 scenario (new pattern): 3 aug models only
K7=7
K7_CLEAN_TIME_1="20260112_10:01:45"
K7_CLEAN_TIME_2="20260112_18:10:41"

# k5clean4 scenario (new pattern): 3 aug models only
K5=5
K5_CLEAN_TIME_1="20260112_10:01:45"
K5_CLEAN_TIME_2="20260112_11:02:15"
K5_CLEAN_TIME_3="20260112_17:11:48"
K5_CLEAN_TIME_4="20260112_18:10:41"

# k3clean6 scenario (new pattern): 3 aug models only
K3=3
K3_CLEAN_TIME_1="20260112_10:01:45"
K3_CLEAN_TIME_2="20260112_11:02:15"
K3_CLEAN_TIME_3="20260112_12:03:40"
K3_CLEAN_TIME_4="20260112_16:15:51"
K3_CLEAN_TIME_5="20260112_17:11:48"
K3_CLEAN_TIME_6="20260112_18:10:41"

# scenario: base4 | k7clean2 | k5clean4 | k3clean6 | both
SCENARIO="base4"

usage() {
  cat <<'EOF'
Usage:
  run_make_4models.sh [options]

Options:
  --scenario base4|k7clean2|k5clean4|k3clean6|both  (default: base4)
  --time YYYYMMDD_HH:MM:SS         (default: 20260112_14:08:03)

  # base4 parameters
  --k N                            (default: 9)
  --rand-n N                       (default: 2)
  --rand-m N                       (default: 9)
  --rand-p P                       (default: 0.5)

  # k7clean2 parameters
  --k7 N                           (default: 7)
  --k7-clean-time1 TIME            (default: 20260112_10:01:45)
  --k7-clean-time2 TIME            (default: 20260112_18:10:41)

  # k5clean4 parameters
  --k5 N                           (default: 5)
  --k5-clean-time1 TIME            (default: 20260112_10:01:45)
  --k5-clean-time2 TIME            (default: 20260112_11:02:15)
  --k5-clean-time3 TIME            (default: 20260112_17:11:48)
  --k5-clean-time4 TIME            (default: 20260112_18:10:41)

  # k3clean6 parameters
  --k3 N                           (default: 3)
  --k3-clean-time1 TIME            (default: 20260112_10:01:45)
  --k3-clean-time2 TIME            (default: 20260112_11:02:15)
  --k3-clean-time3 TIME            (default: 20260112_12:03:40)
  --k3-clean-time4 TIME            (default: 20260112_16:15:51)
  --k3-clean-time5 TIME            (default: 20260112_17:11:48)
  --k3-clean-time6 TIME            (default: 20260112_18:10:41)

Scenarios:
  base4:
    - Generate K augmentations from --time
    - Train 4 models: baseline + augmix + TrivialAugmentWide + rand_augment
    - baseline uses fixed non-aug clean times:
        20260112_10:01:45, 20260112_11:02:15, 20260112_12:03:40,
        20260112_13:06:37, 20260112_14:08:03, 20260112_15:03:18,
        20260112_16:15:51, 20260112_17:11:48, 20260112_18:10:41

  k7clean2:
    - Generate K7 augmentations from --time
    - Build mixed training sets by combining:
        (augmented --time dataset) + (non-aug clean-time1) + (non-aug clean-time2)
    - Train 3 models: augmix + TrivialAugmentWide + rand_augment

  k5clean4:
    - Generate K5 augmentations from --time
    - Build mixed training sets by combining:
        (augmented --time dataset) + 4 non-aug clean times
    - Train 3 models: augmix + TrivialAugmentWide + rand_augment

  k3clean6:
    - Generate K3 augmentations from --time
    - Build mixed training sets by combining:
        (augmented --time dataset) + 6 non-aug clean times
    - Train 3 models: augmix + TrivialAugmentWide + rand_augment

  both:
    - Run base4, then run k7clean2

config/config.yaml is temporarily overwritten and restored automatically.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenario)
      SCENARIO="${2:?missing value for --scenario}"
      shift 2
      ;;
    --time)
      TARGET_TIME="${2:?missing value for --time}"
      shift 2
      ;;
    --k)
      K="${2:?missing value for --k}"
      shift 2
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
    --k7)
      K7="${2:?missing value for --k7}"
      shift 2
      ;;
    --k7-clean-time1)
      K7_CLEAN_TIME_1="${2:?missing value for --k7-clean-time1}"
      shift 2
      ;;
    --k7-clean-time2)
      K7_CLEAN_TIME_2="${2:?missing value for --k7-clean-time2}"
      shift 2
      ;;
    --k5)
      K5="${2:?missing value for --k5}"
      shift 2
      ;;
    --k5-clean-time1)
      K5_CLEAN_TIME_1="${2:?missing value for --k5-clean-time1}"
      shift 2
      ;;
    --k5-clean-time2)
      K5_CLEAN_TIME_2="${2:?missing value for --k5-clean-time2}"
      shift 2
      ;;
    --k5-clean-time3)
      K5_CLEAN_TIME_3="${2:?missing value for --k5-clean-time3}"
      shift 2
      ;;
    --k5-clean-time4)
      K5_CLEAN_TIME_4="${2:?missing value for --k5-clean-time4}"
      shift 2
      ;;
    --k3)
      K3="${2:?missing value for --k3}"
      shift 2
      ;;
    --k3-clean-time1)
      K3_CLEAN_TIME_1="${2:?missing value for --k3-clean-time1}"
      shift 2
      ;;
    --k3-clean-time2)
      K3_CLEAN_TIME_2="${2:?missing value for --k3-clean-time2}"
      shift 2
      ;;
    --k3-clean-time3)
      K3_CLEAN_TIME_3="${2:?missing value for --k3-clean-time3}"
      shift 2
      ;;
    --k3-clean-time4)
      K3_CLEAN_TIME_4="${2:?missing value for --k3-clean-time4}"
      shift 2
      ;;
    --k3-clean-time5)
      K3_CLEAN_TIME_5="${2:?missing value for --k3-clean-time5}"
      shift 2
      ;;
    --k3-clean-time6)
      K3_CLEAN_TIME_6="${2:?missing value for --k3-clean-time6}"
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

case "$SCENARIO" in
  base4|k7clean2|k5clean4|k3clean6|both) ;;
  *)
    echo "[ERR] invalid --scenario: $SCENARIO" >&2
    usage
    exit 1
    ;;
esac

for f in "$CFG" "$AUGMIX_PY" "$TAW_PY" "$RANDAUG_PY" "$LEARN_PY"; do
  [[ -f "$f" ]] || { echo "[ERR] missing file: $f" >&2; exit 1; }
done

check_dataset_exists() {
  local time_id="$1"
  local ds="$NAV_DIR/data/$time_id/dataset"
  [[ -d "$ds/img" ]] || { echo "[ERR] missing: $ds/img" >&2; exit 1; }
  [[ -f "$ds/vel/data.csv" ]] || { echo "[ERR] missing: $ds/vel/data.csv" >&2; exit 1; }
}

check_dataset_exists "$TARGET_TIME"
if [[ "$SCENARIO" == "base4" || "$SCENARIO" == "both" ]]; then
  for t in "${BASELINE_CLEAN_TIMES[@]}"; do
    check_dataset_exists "$t"
  done
fi
if [[ "$SCENARIO" == "k7clean2" || "$SCENARIO" == "both" ]]; then
  check_dataset_exists "$K7_CLEAN_TIME_1"
  check_dataset_exists "$K7_CLEAN_TIME_2"
fi
if [[ "$SCENARIO" == "k5clean4" ]]; then
  check_dataset_exists "$K5_CLEAN_TIME_1"
  check_dataset_exists "$K5_CLEAN_TIME_2"
  check_dataset_exists "$K5_CLEAN_TIME_3"
  check_dataset_exists "$K5_CLEAN_TIME_4"
fi
if [[ "$SCENARIO" == "k3clean6" ]]; then
  check_dataset_exists "$K3_CLEAN_TIME_1"
  check_dataset_exists "$K3_CLEAN_TIME_2"
  check_dataset_exists "$K3_CLEAN_TIME_3"
  check_dataset_exists "$K3_CLEAN_TIME_4"
  check_dataset_exists "$K3_CLEAN_TIME_5"
  check_dataset_exists "$K3_CLEAN_TIME_6"
fi

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

run_aug() {
  local name="$1"
  local py="$2"
  local out_img="$3"
  local out_vel="$4"
  local k_value="$5"
  shift 5
  echo "[RUN] $name augmentation (time=$TARGET_TIME, K=$k_value)"
  set_cfg_for_aug "img" "vel" "$out_img" "$out_vel"
  "$PYTHON_BIN" "$py" --K "$k_value" "$@"
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

build_mixed_dataset() {
  local tag="$1"
  local aug_img="$2"
  local aug_vel="$3"
  local out_img="$4"
  local out_vel="$5"
  shift 5
  local clean_times=("$@")

  if [[ ${#clean_times[@]} -eq 0 ]]; then
    echo "[ERR] build_mixed_dataset requires at least one clean time" >&2
    exit 1
  fi

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

sources = [
    ("aug", target, aug_img, aug_vel),
]
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

build_clean_only_dataset() {
  local tag="$1"
  local out_img="$2"
  local out_vel="$3"
  shift 3
  local clean_times=("$@")

  if [[ ${#clean_times[@]} -eq 0 ]]; then
    echo "[ERR] build_clean_only_dataset requires at least one clean time" >&2
    exit 1
  fi

  local clean_csv
  clean_csv="$(IFS=,; echo "${clean_times[*]}")"
  echo "[RUN] build clean-only dataset ($tag): clean times = $clean_csv"

  NAV_PATH="$NAV_DIR" TARGET_TIME="$TARGET_TIME" CLEAN_TIMES_CSV="$clean_csv" OUT_IMG="$out_img" OUT_VEL="$out_vel" "$PYTHON_BIN" - <<'PY'
import os
import shutil
import pandas as pd

nav = os.environ["NAV_PATH"]
data_root = os.path.join(nav, "data")
target = os.environ["TARGET_TIME"]
clean_times = [x for x in os.environ["CLEAN_TIMES_CSV"].split(",") if x]
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

rows = []
missing_rows = 0
for i, time_id in enumerate(clean_times, start=1):
    ds_root = os.path.join(data_root, time_id, "dataset")
    img_root = os.path.join(ds_root, "img")
    csv_path = os.path.join(ds_root, "vel", "data.csv")

    if not os.path.isdir(img_root):
        raise FileNotFoundError(f"img dir not found: {img_root}")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "episode" not in df.columns:
        raise KeyError(f"'episode' column not found: {csv_path}")

    safe_time = time_id.replace(":", "-")
    label = f"clean{i}"
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
print(f"[DONE] clean-only rows={len(rows)} missing_rows={missing_rows}")
print(f"[DONE] clean-only img dir: {out_img_dir}")
print(f"[DONE] clean-only csv    : {out_csv}")
PY
}

run_base4() {
  local safe_target="${TARGET_TIME//:/-}"
  local base_dataset_dir="$NAV_DIR/data/$TARGET_TIME/dataset"
  local model_dir="$NAV_DIR/data/$TARGET_TIME/model/$EPOCH"

  local augmix_img="augmix_img_K${K}"
  local augmix_vel="augmix_vel_K${K}"
  local taw_img="TrivialAugmentWide_img_K${K}"
  local taw_vel="TrivialAugmentWide_vel_K${K}"
  local rand_img="randaugment_img_K${K}"
  local rand_vel="randaugment_vel_K${K}"
  local baseline_mix_img="mix_baseline_img_allclean9"
  local baseline_mix_vel="mix_baseline_vel_allclean9"

  local model_baseline="model_gpu_baseline_allclean9_${safe_target}.pt"
  local model_augmix="model_gpu_augmix_K${K}_${safe_target}.pt"
  local model_taw="model_gpu_trivialaugwide_K${K}_${safe_target}.pt"
  local model_rand="model_gpu_randaugment_K${K}_${safe_target}.pt"

  echo "[SCENARIO] base4"
  run_aug "AugMix" "$AUGMIX_PY" "$augmix_img" "$augmix_vel" "$K"
  run_aug "TrivialAugmentWide" "$TAW_PY" "$taw_img" "$taw_vel" "$K" --same3view
  run_aug "RandAugment" "$RANDAUG_PY" "$rand_img" "$rand_vel" "$K" --N "$RAND_N" --M "$RAND_M" --p "$RAND_P" --sync_views

  build_clean_only_dataset "baseline_clean9" "$baseline_mix_img" "$baseline_mix_vel" "${BASELINE_CLEAN_TIMES[@]}"

  run_train "baseline(no-aug clean9)" "$baseline_mix_img" "$baseline_mix_vel" "$model_baseline"
  run_train "augmix" "$augmix_img" "$augmix_vel" "$model_augmix"
  run_train "TrivialAugmentWide" "$taw_img" "$taw_vel" "$model_taw"
  run_train "rand_augment" "$rand_img" "$rand_vel" "$model_rand"

  echo "[DONE][base4] models:"
  echo "  $model_dir/$model_baseline"
  echo "  $model_dir/$model_augmix"
  echo "  $model_dir/$model_taw"
  echo "  $model_dir/$model_rand"
  echo "[DONE][base4] augmented datasets:"
  echo "  $base_dataset_dir/$augmix_img"
  echo "  $base_dataset_dir/$taw_img"
  echo "  $base_dataset_dir/$rand_img"
  echo "[DONE][base4] baseline mixed training dataset:"
  echo "  $base_dataset_dir/$baseline_mix_img + $base_dataset_dir/$baseline_mix_vel"
}

run_k7clean2() {
  local safe_target="${TARGET_TIME//:/-}"
  local safe_c1="${K7_CLEAN_TIME_1//:/-}"
  local safe_c2="${K7_CLEAN_TIME_2//:/-}"
  local base_dataset_dir="$NAV_DIR/data/$TARGET_TIME/dataset"
  local model_dir="$NAV_DIR/data/$TARGET_TIME/model/$EPOCH"

  local augmix_img="augmix_img_K${K7}"
  local augmix_vel="augmix_vel_K${K7}"
  local taw_img="TrivialAugmentWide_img_K${K7}"
  local taw_vel="TrivialAugmentWide_vel_K${K7}"
  local rand_img="randaugment_img_K${K7}"
  local rand_vel="randaugment_vel_K${K7}"

  local mix_augmix_img="mix_augmix_img_K${K7}_plus2clean"
  local mix_augmix_vel="mix_augmix_vel_K${K7}_plus2clean"
  local mix_taw_img="mix_taw_img_K${K7}_plus2clean"
  local mix_taw_vel="mix_taw_vel_K${K7}_plus2clean"
  local mix_rand_img="mix_randaugment_img_K${K7}_plus2clean"
  local mix_rand_vel="mix_randaugment_vel_K${K7}_plus2clean"

  local model_augmix="model_gpu_augmix_K${K7}_plusclean2_${safe_target}_${safe_c1}_${safe_c2}.pt"
  local model_taw="model_gpu_trivialaugwide_K${K7}_plusclean2_${safe_target}_${safe_c1}_${safe_c2}.pt"
  local model_rand="model_gpu_randaugment_K${K7}_plusclean2_${safe_target}_${safe_c1}_${safe_c2}.pt"

  echo "[SCENARIO] k7clean2"
  echo "[INFO] clean datasets: $K7_CLEAN_TIME_1, $K7_CLEAN_TIME_2"

  run_aug "AugMix" "$AUGMIX_PY" "$augmix_img" "$augmix_vel" "$K7"
  run_aug "TrivialAugmentWide" "$TAW_PY" "$taw_img" "$taw_vel" "$K7" --same3view
  run_aug "RandAugment" "$RANDAUG_PY" "$rand_img" "$rand_vel" "$K7" --N "$RAND_N" --M "$RAND_M" --p "$RAND_P" --sync_views

  build_mixed_dataset "augmix" "$augmix_img" "$augmix_vel" "$mix_augmix_img" "$mix_augmix_vel" \
    "$K7_CLEAN_TIME_1" "$K7_CLEAN_TIME_2"
  build_mixed_dataset "TrivialAugmentWide" "$taw_img" "$taw_vel" "$mix_taw_img" "$mix_taw_vel" \
    "$K7_CLEAN_TIME_1" "$K7_CLEAN_TIME_2"
  build_mixed_dataset "rand_augment" "$rand_img" "$rand_vel" "$mix_rand_img" "$mix_rand_vel" \
    "$K7_CLEAN_TIME_1" "$K7_CLEAN_TIME_2"

  run_train "augmix(K7 + clean2)" "$mix_augmix_img" "$mix_augmix_vel" "$model_augmix"
  run_train "TrivialAugmentWide(K7 + clean2)" "$mix_taw_img" "$mix_taw_vel" "$model_taw"
  run_train "rand_augment(K7 + clean2)" "$mix_rand_img" "$mix_rand_vel" "$model_rand"

  echo "[DONE][k7clean2] models:"
  echo "  $model_dir/$model_augmix"
  echo "  $model_dir/$model_taw"
  echo "  $model_dir/$model_rand"
  echo "[DONE][k7clean2] augmented datasets:"
  echo "  $base_dataset_dir/$augmix_img"
  echo "  $base_dataset_dir/$taw_img"
  echo "  $base_dataset_dir/$rand_img"
  echo "[DONE][k7clean2] mixed training datasets:"
  echo "  $base_dataset_dir/$mix_augmix_img + $base_dataset_dir/$mix_augmix_vel"
  echo "  $base_dataset_dir/$mix_taw_img + $base_dataset_dir/$mix_taw_vel"
  echo "  $base_dataset_dir/$mix_rand_img + $base_dataset_dir/$mix_rand_vel"
}

run_k5clean4() {
  local safe_target="${TARGET_TIME//:/-}"
  local safe_c1="${K5_CLEAN_TIME_1//:/-}"
  local safe_c2="${K5_CLEAN_TIME_2//:/-}"
  local safe_c3="${K5_CLEAN_TIME_3//:/-}"
  local safe_c4="${K5_CLEAN_TIME_4//:/-}"
  local base_dataset_dir="$NAV_DIR/data/$TARGET_TIME/dataset"
  local model_dir="$NAV_DIR/data/$TARGET_TIME/model/$EPOCH"

  local augmix_img="augmix_img_K${K5}"
  local augmix_vel="augmix_vel_K${K5}"
  local taw_img="TrivialAugmentWide_img_K${K5}"
  local taw_vel="TrivialAugmentWide_vel_K${K5}"
  local rand_img="randaugment_img_K${K5}"
  local rand_vel="randaugment_vel_K${K5}"

  local mix_augmix_img="mix_augmix_img_K${K5}_plus4clean"
  local mix_augmix_vel="mix_augmix_vel_K${K5}_plus4clean"
  local mix_taw_img="mix_taw_img_K${K5}_plus4clean"
  local mix_taw_vel="mix_taw_vel_K${K5}_plus4clean"
  local mix_rand_img="mix_randaugment_img_K${K5}_plus4clean"
  local mix_rand_vel="mix_randaugment_vel_K${K5}_plus4clean"

  local model_augmix="model_gpu_augmix_K${K5}_plusclean4_${safe_target}_${safe_c1}_${safe_c2}_${safe_c3}_${safe_c4}.pt"
  local model_taw="model_gpu_trivialaugwide_K${K5}_plusclean4_${safe_target}_${safe_c1}_${safe_c2}_${safe_c3}_${safe_c4}.pt"
  local model_rand="model_gpu_randaugment_K${K5}_plusclean4_${safe_target}_${safe_c1}_${safe_c2}_${safe_c3}_${safe_c4}.pt"

  echo "[SCENARIO] k5clean4"
  echo "[INFO] clean datasets: $K5_CLEAN_TIME_1, $K5_CLEAN_TIME_2, $K5_CLEAN_TIME_3, $K5_CLEAN_TIME_4"

  run_aug "AugMix" "$AUGMIX_PY" "$augmix_img" "$augmix_vel" "$K5"
  run_aug "TrivialAugmentWide" "$TAW_PY" "$taw_img" "$taw_vel" "$K5" --same3view
  run_aug "RandAugment" "$RANDAUG_PY" "$rand_img" "$rand_vel" "$K5" --N "$RAND_N" --M "$RAND_M" --p "$RAND_P" --sync_views

  build_mixed_dataset "augmix" "$augmix_img" "$augmix_vel" "$mix_augmix_img" "$mix_augmix_vel" \
    "$K5_CLEAN_TIME_1" "$K5_CLEAN_TIME_2" "$K5_CLEAN_TIME_3" "$K5_CLEAN_TIME_4"
  build_mixed_dataset "TrivialAugmentWide" "$taw_img" "$taw_vel" "$mix_taw_img" "$mix_taw_vel" \
    "$K5_CLEAN_TIME_1" "$K5_CLEAN_TIME_2" "$K5_CLEAN_TIME_3" "$K5_CLEAN_TIME_4"
  build_mixed_dataset "rand_augment" "$rand_img" "$rand_vel" "$mix_rand_img" "$mix_rand_vel" \
    "$K5_CLEAN_TIME_1" "$K5_CLEAN_TIME_2" "$K5_CLEAN_TIME_3" "$K5_CLEAN_TIME_4"

  run_train "augmix(K5 + clean4)" "$mix_augmix_img" "$mix_augmix_vel" "$model_augmix"
  run_train "TrivialAugmentWide(K5 + clean4)" "$mix_taw_img" "$mix_taw_vel" "$model_taw"
  run_train "rand_augment(K5 + clean4)" "$mix_rand_img" "$mix_rand_vel" "$model_rand"

  echo "[DONE][k5clean4] models:"
  echo "  $model_dir/$model_augmix"
  echo "  $model_dir/$model_taw"
  echo "  $model_dir/$model_rand"
  echo "[DONE][k5clean4] augmented datasets:"
  echo "  $base_dataset_dir/$augmix_img"
  echo "  $base_dataset_dir/$taw_img"
  echo "  $base_dataset_dir/$rand_img"
  echo "[DONE][k5clean4] mixed training datasets:"
  echo "  $base_dataset_dir/$mix_augmix_img + $base_dataset_dir/$mix_augmix_vel"
  echo "  $base_dataset_dir/$mix_taw_img + $base_dataset_dir/$mix_taw_vel"
  echo "  $base_dataset_dir/$mix_rand_img + $base_dataset_dir/$mix_rand_vel"
}

run_k3clean6() {
  local safe_target="${TARGET_TIME//:/-}"
  local safe_c1="${K3_CLEAN_TIME_1//:/-}"
  local safe_c2="${K3_CLEAN_TIME_2//:/-}"
  local safe_c3="${K3_CLEAN_TIME_3//:/-}"
  local safe_c4="${K3_CLEAN_TIME_4//:/-}"
  local safe_c5="${K3_CLEAN_TIME_5//:/-}"
  local safe_c6="${K3_CLEAN_TIME_6//:/-}"
  local base_dataset_dir="$NAV_DIR/data/$TARGET_TIME/dataset"
  local model_dir="$NAV_DIR/data/$TARGET_TIME/model/$EPOCH"

  local augmix_img="augmix_img_K${K3}"
  local augmix_vel="augmix_vel_K${K3}"
  local taw_img="TrivialAugmentWide_img_K${K3}"
  local taw_vel="TrivialAugmentWide_vel_K${K3}"
  local rand_img="randaugment_img_K${K3}"
  local rand_vel="randaugment_vel_K${K3}"

  local mix_augmix_img="mix_augmix_img_K${K3}_plus6clean"
  local mix_augmix_vel="mix_augmix_vel_K${K3}_plus6clean"
  local mix_taw_img="mix_taw_img_K${K3}_plus6clean"
  local mix_taw_vel="mix_taw_vel_K${K3}_plus6clean"
  local mix_rand_img="mix_randaugment_img_K${K3}_plus6clean"
  local mix_rand_vel="mix_randaugment_vel_K${K3}_plus6clean"

  local model_augmix="model_gpu_augmix_K${K3}_plusclean6_${safe_target}_${safe_c1}_${safe_c2}_${safe_c3}_${safe_c4}_${safe_c5}_${safe_c6}.pt"
  local model_taw="model_gpu_trivialaugwide_K${K3}_plusclean6_${safe_target}_${safe_c1}_${safe_c2}_${safe_c3}_${safe_c4}_${safe_c5}_${safe_c6}.pt"
  local model_rand="model_gpu_randaugment_K${K3}_plusclean6_${safe_target}_${safe_c1}_${safe_c2}_${safe_c3}_${safe_c4}_${safe_c5}_${safe_c6}.pt"

  echo "[SCENARIO] k3clean6"
  echo "[INFO] clean datasets: $K3_CLEAN_TIME_1, $K3_CLEAN_TIME_2, $K3_CLEAN_TIME_3, $K3_CLEAN_TIME_4, $K3_CLEAN_TIME_5, $K3_CLEAN_TIME_6"

  run_aug "AugMix" "$AUGMIX_PY" "$augmix_img" "$augmix_vel" "$K3"
  run_aug "TrivialAugmentWide" "$TAW_PY" "$taw_img" "$taw_vel" "$K3" --same3view
  run_aug "RandAugment" "$RANDAUG_PY" "$rand_img" "$rand_vel" "$K3" --N "$RAND_N" --M "$RAND_M" --p "$RAND_P" --sync_views

  build_mixed_dataset "augmix" "$augmix_img" "$augmix_vel" "$mix_augmix_img" "$mix_augmix_vel" \
    "$K3_CLEAN_TIME_1" "$K3_CLEAN_TIME_2" "$K3_CLEAN_TIME_3" "$K3_CLEAN_TIME_4" "$K3_CLEAN_TIME_5" "$K3_CLEAN_TIME_6"
  build_mixed_dataset "TrivialAugmentWide" "$taw_img" "$taw_vel" "$mix_taw_img" "$mix_taw_vel" \
    "$K3_CLEAN_TIME_1" "$K3_CLEAN_TIME_2" "$K3_CLEAN_TIME_3" "$K3_CLEAN_TIME_4" "$K3_CLEAN_TIME_5" "$K3_CLEAN_TIME_6"
  build_mixed_dataset "rand_augment" "$rand_img" "$rand_vel" "$mix_rand_img" "$mix_rand_vel" \
    "$K3_CLEAN_TIME_1" "$K3_CLEAN_TIME_2" "$K3_CLEAN_TIME_3" "$K3_CLEAN_TIME_4" "$K3_CLEAN_TIME_5" "$K3_CLEAN_TIME_6"

  run_train "augmix(K3 + clean6)" "$mix_augmix_img" "$mix_augmix_vel" "$model_augmix"
  run_train "TrivialAugmentWide(K3 + clean6)" "$mix_taw_img" "$mix_taw_vel" "$model_taw"
  run_train "rand_augment(K3 + clean6)" "$mix_rand_img" "$mix_rand_vel" "$model_rand"

  echo "[DONE][k3clean6] models:"
  echo "  $model_dir/$model_augmix"
  echo "  $model_dir/$model_taw"
  echo "  $model_dir/$model_rand"
  echo "[DONE][k3clean6] augmented datasets:"
  echo "  $base_dataset_dir/$augmix_img"
  echo "  $base_dataset_dir/$taw_img"
  echo "  $base_dataset_dir/$rand_img"
  echo "[DONE][k3clean6] mixed training datasets:"
  echo "  $base_dataset_dir/$mix_augmix_img + $base_dataset_dir/$mix_augmix_vel"
  echo "  $base_dataset_dir/$mix_taw_img + $base_dataset_dir/$mix_taw_vel"
  echo "  $base_dataset_dir/$mix_rand_img + $base_dataset_dir/$mix_rand_vel"
}

echo "[INFO] scenario: $SCENARIO"
echo "[INFO] target time: $TARGET_TIME"
echo "[INFO] epoch (from config): $EPOCH"

if [[ "$SCENARIO" == "base4" || "$SCENARIO" == "both" ]]; then
  run_base4
fi

if [[ "$SCENARIO" == "k7clean2" || "$SCENARIO" == "both" ]]; then
  run_k7clean2
fi

if [[ "$SCENARIO" == "k5clean4" ]]; then
  run_k5clean4
fi

if [[ "$SCENARIO" == "k3clean6" ]]; then
  run_k3clean6
fi

echo "[DONE] scenario completed: $SCENARIO"
