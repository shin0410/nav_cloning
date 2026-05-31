#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$NAV_DIR/data"
CFG="$NAV_DIR/config/config.yaml"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SPLIT_PY="$NAV_DIR/scripts/split_dataset_first_n.py"
LEARN_PY="$NAV_DIR/scripts/learning_surprise.py"
TAW_PY="$NAV_DIR/scripts/edit_dataset/aug_TrivialAugmentWide.py"
EVAL_PY="$NAV_DIR/scripts/eval_baseline_hour_matrix.py"
PLOT_PY="$NAV_DIR/scripts/plot_baseline_hour_matrix.py"
ANALYZE_GAP_PY="$NAV_DIR/scripts/analyze_baseline_lux_brightness_gap.py"

TRAIN_ROWS=4000
TRAIN_IMG_DIR="train4000_img"
TRAIN_VEL_DIR="train4000_vel"
TEST_IMG_DIR="test_rest_img"
TEST_VEL_DIR="test_rest_vel"
OUTPUT_TIME="_taw_hour_matrix_20260308_0310"
OUT_DIR=""
RUN_UNZIP=1
RUN_SPLIT=1
RUN_TRAIN=1
RUN_PLOT=1
RUN_ANALYZE=1
RUN_BASELINE_COMPARE=1

K=5
MAG=31
OPS_CSV=""
MAG_BINS_CSV=""
LIST_OPS=0
BASELINE_EVAL_DIR="$DATA_DIR/_baseline_hour_matrix_20260308_0310/eval"

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
  run_hour_matrix_taw_202603.sh [options]

Options:
  --train-rows N            default: 4000
  --output-time NAME        default: _taw_hour_matrix_20260308_0310
  --out-dir DIR             default: data/<output-time>
  --k N                     TrivialAugmentWide K (default: 5)
  --mag N                   magnitude bins for default/single-op run (default: 31)
  --ops LIST                single-op sweep, e.g. Brightness,Contrast,Sharpness
  --mag-bins LIST           strength sweep, e.g. 31,25,20,15,10,5
  --list-ops                print available ops for current --mag and exit
  --skip-unzip              do not extract zip files
  --skip-split              do not rebuild train/test split dirs
  --skip-train              skip retraining and re-run eval only
  --skip-plot               skip heatmap/bar plots
  --skip-analyze            skip lux/brightness gap analysis
  --baseline-eval-dir DIR   default: data/_baseline_hour_matrix_20260308_0310/eval
  --skip-baseline-compare   skip delta-vs-baseline export
  -h, --help

Modes:
  default:
    no --ops and no --mag-bins
    -> train one TAW model per hour with all allowed ops and M=--mag

  single-op sweep:
    --ops AutoContrast,Brightness,...
    -> for each op, train 9 TAW models (one per hour) and export one 9x9 eval

  strength sweep:
    --mag-bins 31,25,20,15,10,5
    -> for each magnitude bin count, train 9 TAW models and export one 9x9 eval

Notes:
  - --ops and --mag-bins are mutually exclusive in one run.
  - All TAW runs use --same3view.
  - Eval outputs are stored per variant under data/<output-time>/eval/<variant>/.
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
    --k)
      K="${2:?missing value for --k}"
      shift 2
      ;;
    --mag)
      MAG="${2:?missing value for --mag}"
      shift 2
      ;;
    --ops)
      OPS_CSV="${2:?missing value for --ops}"
      shift 2
      ;;
    --mag-bins)
      MAG_BINS_CSV="${2:?missing value for --mag-bins}"
      shift 2
      ;;
    --list-ops)
      LIST_OPS=1
      shift
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
    --skip-analyze)
      RUN_ANALYZE=0
      shift
      ;;
    --baseline-eval-dir)
      BASELINE_EVAL_DIR="${2:?missing value for --baseline-eval-dir}"
      shift 2
      ;;
    --skip-baseline-compare)
      RUN_BASELINE_COMPARE=0
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

for f in "$CFG" "$SPLIT_PY" "$LEARN_PY" "$TAW_PY" "$EVAL_PY" "$PLOT_PY" "$ANALYZE_GAP_PY"; do
  [[ -f "$f" ]] || { echo "[ERR] missing file: $f" >&2; exit 1; }
done

[[ "$K" =~ ^[0-9]+$ ]] || { echo "[ERR] --k must be integer: $K" >&2; exit 1; }
(( K >= 1 )) || { echo "[ERR] --k must be >= 1" >&2; exit 1; }
[[ "$MAG" =~ ^[0-9]+$ ]] || { echo "[ERR] --mag must be integer: $MAG" >&2; exit 1; }
(( MAG >= 2 )) || { echo "[ERR] --mag must be >= 2" >&2; exit 1; }

if [[ -n "$OPS_CSV" && -n "$MAG_BINS_CSV" ]]; then
  echo "[ERR] --ops and --mag-bins cannot be used together" >&2
  exit 1
fi

if [[ "$LIST_OPS" -eq 1 ]]; then
  "$PYTHON_BIN" "$TAW_PY" --num-magnitude-bins "$MAG" --list-ops
  exit 0
fi

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$DATA_DIR/$OUTPUT_TIME"
fi
mkdir -p "$OUT_DIR"

parse_csv_list() {
  local csv="$1"
  local kind="$2"
  local -n out_ref="$3"
  local -a parsed=()
  declare -A seen=()

  IFS=',' read -r -a tokens <<< "$csv"
  for token in "${tokens[@]}"; do
    local v="${token#"${token%%[![:space:]]*}"}"
    v="${v%"${v##*[![:space:]]}"}"
    [[ -n "$v" ]] || continue
    if [[ "$kind" == "int" ]]; then
      [[ "$v" =~ ^[0-9]+$ ]] || { echo "[ERR] invalid integer token: $token" >&2; exit 1; }
      (( v >= 2 )) || { echo "[ERR] magnitude must be >= 2: $v" >&2; exit 1; }
    fi
    if [[ -z "${seen[$v]+x}" ]]; then
      seen[$v]=1
      parsed+=("$v")
    fi
  done

  [[ ${#parsed[@]} -gt 0 ]] || { echo "[ERR] no valid values in CSV: $csv" >&2; exit 1; }
  out_ref=("${parsed[@]}")
}

OPS=()
MAG_BINS=()
if [[ -n "$OPS_CSV" ]]; then
  parse_csv_list "$OPS_CSV" "str" OPS
  AVAILABLE_OPS_RAW="$("$PYTHON_BIN" "$TAW_PY" --num-magnitude-bins "$MAG" --list-ops)"
  declare -A AVAIL=()
  IFS=',' read -r -a _avail_tokens <<< "$AVAILABLE_OPS_RAW"
  for op in "${_avail_tokens[@]}"; do
    AVAIL["$op"]=1
  done
  for op in "${OPS[@]}"; do
    [[ -n "${AVAIL[$op]+x}" ]] || {
      echo "[ERR] unsupported op '$op' for --mag=$MAG" >&2
      echo "      available: $AVAILABLE_OPS_RAW" >&2
      exit 1
    }
  done
fi

if [[ -n "$MAG_BINS_CSV" ]]; then
  parse_csv_list "$MAG_BINS_CSV" "int" MAG_BINS
fi

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

extract_hour() {
  local time_id="$1"
  echo "${time_id#*_}" | cut -c1-2
}

op_safe_name() {
  local op="$1"
  echo "$op" | tr -c 'A-Za-z0-9_' '_'
}

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

run_taw_aug() {
  local time_id="$1"
  local out_img="$2"
  local out_vel="$3"
  local mag="$4"
  local allowed_ops="${5:-}"

  echo "[RUN] TAW augmentation time=$time_id K=$K M=$mag allowed_ops=${allowed_ops:-ALL}"
  set_cfg_for_aug "$time_id" "$TRAIN_IMG_DIR" "$TRAIN_VEL_DIR" "$out_img" "$out_vel"

  local cmd=(
    "$TAW_PY"
    "--K" "$K"
    "--same3view"
    "--num-magnitude-bins" "$mag"
  )
  if [[ -n "$allowed_ops" ]]; then
    cmd+=("--allowed-ops" "$allowed_ops")
  fi
  "$PYTHON_BIN" "${cmd[@]}"
}

run_train_model() {
  local time_id="$1"
  local label="$2"
  local load_img="$3"
  local load_vel="$4"
  local save_model="$5"
  echo "[RUN] train $label time=$time_id save=$save_model"
  NAV_TIME="$time_id" \
  NAV_TRAIN_TIMES="$time_id" \
  NAV_OUTPUT_TIME="$OUTPUT_TIME" \
  NAV_LOAD_DATASET_IMG="$load_img" \
  NAV_LOAD_DATASET_VEL="$load_vel" \
  NAV_SAVE_MODEL="$save_model" \
  "$PYTHON_BIN" "$LEARN_PY"
}

append_variant_summary() {
  local variant_id="$1"
  local sweep_mode="$2"
  local op_name="$3"
  local mag="$4"
  local eval_dir="$5"

  VARIANT_ID="$variant_id" SWEEP_MODE="$sweep_mode" OP_NAME="$op_name" MAG="$mag" EVAL_DIR="$eval_dir" "$PYTHON_BIN" - <<'PY' >> "$OUT_DIR/variant_overview.csv"
import os
from pathlib import Path
import pandas as pd

eval_dir = Path(os.environ["EVAL_DIR"])
df = pd.read_csv(eval_dir / "summary_by_train_test.csv")
same = df.loc[df["same_hour"] == 1, "mae"]
diff = df.loc[df["same_hour"] == 0, "mae"]

cmp_path = eval_dir / "compare_vs_baseline_by_train_test.csv"
delta_mean = ""
delta_same = ""
delta_diff = ""
if cmp_path.is_file():
    cmp_df = pd.read_csv(cmp_path)
    delta_mean = f"{float(cmp_df['mae_delta_vs_baseline'].mean()):.12f}"
    same_cmp = cmp_df.loc[cmp_df["same_hour"] == 1, "mae_delta_vs_baseline"]
    diff_cmp = cmp_df.loc[cmp_df["same_hour"] == 0, "mae_delta_vs_baseline"]
    delta_same = f"{float(same_cmp.mean()):.12f}" if len(same_cmp) else ""
    delta_diff = f"{float(diff_cmp.mean()):.12f}" if len(diff_cmp) else ""

print(
    ",".join(
        [
            os.environ["VARIANT_ID"],
            os.environ["SWEEP_MODE"],
            os.environ["OP_NAME"] or "ALL",
            os.environ["MAG"],
            str(len(df)),
            f"{float(df['mae'].mean()):.12f}",
            f"{float(df['mae'].median()):.12f}",
            f"{float(same.mean()):.12f}" if len(same) else "",
            f"{float(diff.mean()):.12f}" if len(diff) else "",
            delta_mean,
            delta_same,
            delta_diff,
            str(eval_dir),
        ]
    )
)
PY
}

compare_variant_to_baseline() {
  local eval_dir="$1"

  if [[ "$RUN_BASELINE_COMPARE" -ne 1 ]]; then
    echo "[STEP] skip baseline compare (--skip-baseline-compare)"
    return 0
  fi

  if [[ ! -f "$BASELINE_EVAL_DIR/summary_by_train_test.csv" ]]; then
    echo "[WARN] baseline summary not found: $BASELINE_EVAL_DIR/summary_by_train_test.csv"
    return 0
  fi

  echo "[STEP] compare vs baseline eval_dir=$eval_dir"
  VARIANT_EVAL_DIR="$eval_dir" BASELINE_EVAL_DIR="$BASELINE_EVAL_DIR" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
import pandas as pd

eval_dir = Path(os.environ["VARIANT_EVAL_DIR"])
baseline_dir = Path(os.environ["BASELINE_EVAL_DIR"])

variant = pd.read_csv(eval_dir / "summary_by_train_test.csv")
baseline = pd.read_csv(baseline_dir / "summary_by_train_test.csv")

key = ["train_time", "test_time"]
keep_baseline = key + ["mae", "mse", "rmse", "median", "p80", "p95"]
baseline = baseline[keep_baseline].rename(
    columns={
        "mae": "baseline_mae",
        "mse": "baseline_mse",
        "rmse": "baseline_rmse",
        "median": "baseline_median",
        "p80": "baseline_p80",
        "p95": "baseline_p95",
    }
)

merged = variant.merge(baseline, on=key, how="left", validate="one_to_one")
if merged["baseline_mae"].isna().any():
    missing = merged.loc[merged["baseline_mae"].isna(), key]
    raise SystemExit(f"[ERR] missing baseline rows for: {missing.to_dict(orient='records')}")

merged["mae_delta_vs_baseline"] = merged["mae"] - merged["baseline_mae"]
merged["mse_delta_vs_baseline"] = merged["mse"] - merged["baseline_mse"]
merged["rmse_delta_vs_baseline"] = merged["rmse"] - merged["baseline_rmse"]
merged["mae_ratio_vs_baseline"] = merged["mae"] / merged["baseline_mae"]
merged.to_csv(eval_dir / "compare_vs_baseline_by_train_test.csv", index=False)

delta_time = merged.pivot_table(index="train_time", columns="test_time", values="mae_delta_vs_baseline", aggfunc="first")
delta_time.to_csv(eval_dir / "mae_delta_vs_baseline_matrix_by_time.csv")

delta_hour = merged.pivot_table(index="train_hour", columns="test_hour", values="mae_delta_vs_baseline", aggfunc="mean")
delta_hour.to_csv(eval_dir / "mae_delta_vs_baseline_matrix_by_hour.csv")

same_diff = (
    merged.groupby("same_hour", as_index=False)
    .agg(
        n_pairs=("mae_delta_vs_baseline", "size"),
        mean_mae_delta_vs_baseline=("mae_delta_vs_baseline", "mean"),
        median_mae_delta_vs_baseline=("mae_delta_vs_baseline", "median"),
        mean_mae_ratio_vs_baseline=("mae_ratio_vs_baseline", "mean"),
    )
    .sort_values("same_hour")
)
same_diff["same_or_diff"] = same_diff["same_hour"].map({1: "same_hour", 0: "different_hour"})
same_diff.to_csv(eval_dir / "summary_vs_baseline_same_vs_diff.csv", index=False)

hour_gap = (
    merged.groupby("hour_gap", as_index=False)
    .agg(
        n_pairs=("mae_delta_vs_baseline", "size"),
        mean_mae_delta_vs_baseline=("mae_delta_vs_baseline", "mean"),
        median_mae_delta_vs_baseline=("mae_delta_vs_baseline", "median"),
        mean_mae_ratio_vs_baseline=("mae_ratio_vs_baseline", "mean"),
    )
    .sort_values("hour_gap")
)
hour_gap.to_csv(eval_dir / "summary_vs_baseline_by_hour_gap.csv", index=False)

overall = pd.DataFrame(
    [
        {
            "n_pairs": int(len(merged)),
            "mean_mae": float(merged["mae"].mean()),
            "mean_baseline_mae": float(merged["baseline_mae"].mean()),
            "mean_mae_delta_vs_baseline": float(merged["mae_delta_vs_baseline"].mean()),
            "median_mae_delta_vs_baseline": float(merged["mae_delta_vs_baseline"].median()),
            "mean_mae_ratio_vs_baseline": float(merged["mae_ratio_vs_baseline"].mean()),
        }
    ]
)
overall.to_csv(eval_dir / "summary_vs_baseline_overall.csv", index=False)

print(f"[DONE] compare csv   : {eval_dir / 'compare_vs_baseline_by_train_test.csv'}")
print(f"[DONE] delta by time : {eval_dir / 'mae_delta_vs_baseline_matrix_by_time.csv'}")
print(f"[DONE] delta by hour : {eval_dir / 'mae_delta_vs_baseline_matrix_by_hour.csv'}")
print(f"[DONE] overall delta : {eval_dir / 'summary_vs_baseline_overall.csv'}")
PY
}

model_name_for_variant() {
  local time_id="$1"
  local variant_tag="$2"
  local hour
  hour="$(extract_hour "$time_id")"
  echo "model_gpu_trivialaugwide_hour${hour}_${time_id}_${variant_tag}.pt"
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

echo "variant_id,sweep_mode,op_name,magnitude_bins,n_pairs,mean_mae,median_mae,mean_mae_same_hour,mean_mae_diff_hour,mean_mae_delta_vs_baseline,mean_mae_delta_same_hour_vs_baseline,mean_mae_delta_diff_hour_vs_baseline,eval_dir" > "$OUT_DIR/variant_overview.csv"

run_variant() {
  local variant_id="$1"
  local sweep_mode="$2"
  local mag="$3"
  local allowed_ops="$4"

  local safe_variant="$variant_id"
  local aug_img="TrivialAugmentWide_${TRAIN_IMG_DIR}_${safe_variant}"
  local aug_vel="TrivialAugmentWide_${TRAIN_VEL_DIR}_${safe_variant}"
  local variant_tag="K${K}_M${mag}_${safe_variant}"
  local eval_dir="$OUT_DIR/eval/$safe_variant"
  local model_map_csv="$eval_dir/model_map.csv"

  mkdir -p "$eval_dir"
  echo "train_time,train_hour,model_path" > "$model_map_csv"

  if [[ "$RUN_TRAIN" -eq 1 ]]; then
    for t in "${ALL_TIMES[@]}"; do
      local hour model_name
      hour="$(extract_hour "$t")"
      model_name="$(model_name_for_variant "$t" "$variant_tag")"
      run_taw_aug "$t" "$aug_img" "$aug_vel" "$mag" "$allowed_ops"
      run_train_model "$t" "$safe_variant hour=$hour" "$aug_img" "$aug_vel" "$model_name"
      echo "$t,$hour,$MODEL_DIR/$model_name" >> "$model_map_csv"
    done
  else
    echo "[STEP] skip train (--skip-train) variant=$safe_variant"
    for t in "${ALL_TIMES[@]}"; do
      local hour model_name model_path
      hour="$(extract_hour "$t")"
      model_name="$(model_name_for_variant "$t" "$variant_tag")"
      model_path="$MODEL_DIR/$model_name"
      [[ -f "$model_path" ]] || { echo "[ERR] missing model: $model_path" >&2; exit 1; }
      echo "$t,$hour,$model_path" >> "$model_map_csv"
    done
  fi

  echo "[STEP] evaluate variant=$safe_variant"
  "$PYTHON_BIN" "$EVAL_PY" \
    --model-map "$model_map_csv" \
    --data-root "$DATA_DIR" \
    --test-times "${ALL_TIMES[@]}" \
    --test-img-dir "$TEST_IMG_DIR" \
    --test-vel-dir "$TEST_VEL_DIR" \
    --out-dir "$eval_dir"

  if [[ "$RUN_ANALYZE" -eq 1 ]]; then
    echo "[STEP] analyze gaps variant=$safe_variant"
    "$PYTHON_BIN" "$ANALYZE_GAP_PY" \
      --summary-csv "$eval_dir/summary_by_train_test.csv" \
      --data-root "$DATA_DIR" \
      --times "${ALL_TIMES[@]}" \
      --train-img-dir "$aug_img" \
      --train-vel-dir "$aug_vel" \
      --test-img-dir "$TEST_IMG_DIR" \
      --test-vel-dir "$TEST_VEL_DIR" \
      --out-dir "$eval_dir"
  else
    echo "[STEP] skip gap analysis (--skip-analyze) variant=$safe_variant"
  fi

  if [[ "$RUN_PLOT" -eq 1 ]]; then
    echo "[STEP] plot variant=$safe_variant"
    "$PYTHON_BIN" "$PLOT_PY" --eval-dir "$eval_dir"
  else
    echo "[STEP] skip plot (--skip-plot) variant=$safe_variant"
  fi

  compare_variant_to_baseline "$eval_dir"
  append_variant_summary "$safe_variant" "$sweep_mode" "$allowed_ops" "$mag" "$eval_dir"
}

if [[ -n "$OPS_CSV" ]]; then
  echo "[STEP] mode=single-op sweep ops=$(IFS=,; echo "${OPS[*]}")"
  for op in "${OPS[@]}"; do
    safe_op="$(op_safe_name "$op")"
    run_variant "OP${safe_op}_M${MAG}" "single_op" "$MAG" "$op"
  done
elif [[ -n "$MAG_BINS_CSV" ]]; then
  echo "[STEP] mode=strength sweep magnitudes=$(IFS=,; echo "${MAG_BINS[*]}")"
  for mag in "${MAG_BINS[@]}"; do
    run_variant "M${mag}_ALL" "strength" "$mag" ""
  done
else
  echo "[STEP] mode=default all ops, M=$MAG"
  run_variant "M${MAG}_ALL" "default" "$MAG" ""
fi

echo "[DONE] output_time         : $OUTPUT_TIME"
echo "[DONE] model_dir           : $MODEL_DIR"
echo "[DONE] result_dir          : $RESULT_DIR"
echo "[DONE] split_summary.csv   : $OUT_DIR/split_summary.csv"
echo "[DONE] variant overview    : $OUT_DIR/variant_overview.csv"
echo "[DONE] eval root           : $OUT_DIR/eval"
