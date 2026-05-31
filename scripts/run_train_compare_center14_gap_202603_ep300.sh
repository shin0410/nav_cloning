#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="${NAV_DATA_DIR:-/home/shin/challenge_ws/nav_cloning_data}"
OUTPUT_TIME="${NAV_OUTPUT_TIME:-_center14_gap_20260308_0310_ep300}"
EPOCH="${NAV_EPOCH:-300}"

exec "$SCRIPT_DIR/run_train_compare_center14_gap_202603.sh" \
  --data-dir "$DATA_DIR" \
  --output-time "$OUTPUT_TIME" \
  --epoch "$EPOCH" \
  "$@"
