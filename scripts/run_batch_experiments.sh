#!/usr/bin/env bash
set -euo pipefail

# Batch runner for MCGS dual-max, MCGS maternal-only, and MCTS-AHD.
# Required env: KEY, BASE_URL, MODEL_NAME.
# Optional env: JOBS, RUN_ID, PROBLEM_SET, MAX_FE, EXTRA_OVERRIDES.

JOBS="${JOBS:-3}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
PROBLEM_SET="${PROBLEM_SET:-standard}"
MAX_FE="${MAX_FE:-}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
STATUS_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mcgs-batch-status.XXXXXX")"

trap 'rm -rf "$STATUS_DIR"' EXIT

if [[ -z "${KEY:-}" || -z "${BASE_URL:-}" || -z "${MODEL_NAME:-}" ]]; then
  echo "Missing required env. Set KEY, BASE_URL, and MODEL_NAME before running." >&2
  exit 2
fi

case "$PROBLEM_SET" in
  standard)
    PROBLEMS=(
      "tsp_aco"
      "kp_constructive"
      "cvrp_aco"
      "mkp_aco"
      "bpp_offline_aco"
    )
    ;;
  black_box)
    PROBLEMS=(
      "tsp_aco_black_box"
      "kp_constructive"
      "cvrp_aco_black_box"
      "mkp_aco_black_box"
      "bpp_offline_aco_black_box"
    )
    EXTRA_OVERRIDES="init_pop_size=10 ${EXTRA_OVERRIDES}"
    ;;
  *)
    echo "Unsupported PROBLEM_SET=$PROBLEM_SET. Use standard or black_box." >&2
    exit 2
    ;;
esac

VARIANTS=(
  "mcgs_dual_max"
  "mcgs_maternal_only"
  "mcts_ahd"
)

COMMON_OVERRIDES=(
  "llm_client.api_key=$KEY"
  "llm_client.base_url=$BASE_URL"
  "llm_client.model=$MODEL_NAME"
)

if [[ -n "$MAX_FE" ]]; then
  COMMON_OVERRIDES+=("max_fe=$MAX_FE")
fi

if [[ -n "$EXTRA_OVERRIDES" ]]; then
  # Split user-provided Hydra overrides on spaces, matching normal CLI usage.
  read -r -a EXTRA_OVERRIDE_ARRAY <<< "$EXTRA_OVERRIDES"
  COMMON_OVERRIDES+=("${EXTRA_OVERRIDE_ARRAY[@]}")
fi

run_one() {
  local problem="$1"
  local variant="$2"
  local outdir="outputs/batch_${RUN_ID}/${PROBLEM_SET}/${problem}/${variant}"
  local cmd=(python main.py "problem=${problem}" "${COMMON_OVERRIDES[@]}" "hydra.run.dir=${outdir}")

  case "$variant" in
    mcgs_dual_max)
      cmd+=("algorithm=mcgs" "dual_lineage_backup=true" "uct_value_mode=dual_max")
      ;;
    mcgs_maternal_only)
      cmd+=("algorithm=mcgs" "dual_lineage_backup=true" "uct_value_mode=maternal_only")
      ;;
    mcts_ahd)
      cmd+=("algorithm=mcts_ahd")
      ;;
    *)
      echo "Unsupported variant=$variant" >&2
      return 2
      ;;
  esac

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${problem} ${variant} -> ${outdir}"
  "${cmd[@]}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  ${problem} ${variant}"
}

echo "RUN_ID=${RUN_ID}"
echo "PROBLEM_SET=${PROBLEM_SET}"
echo "JOBS=${JOBS}"
echo "Total jobs: $((${#PROBLEMS[@]} * ${#VARIANTS[@]}))"
if [[ "$PROBLEM_SET" == "black_box" ]]; then
  echo "Note: KP uses kp_constructive because this repository has no KP black-box config."
fi

for problem in "${PROBLEMS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    status_file="${STATUS_DIR}/${problem}_${variant}.status"
    (
      if run_one "$problem" "$variant"; then
        echo "ok" > "$status_file"
      else
        echo "failed" > "$status_file"
      fi
    ) &

    while [[ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$JOBS" ]]; do
      sleep 5
    done
  done
done

wait

if grep -Rqx "failed" "$STATUS_DIR"; then
  echo "One or more batch experiments failed:" >&2
  grep -Rxl "failed" "$STATUS_DIR" | sed "s#${STATUS_DIR}/#  - #; s#\\.status\$##" >&2
  exit 1
fi

echo "All batch experiments finished under outputs/batch_${RUN_ID}/${PROBLEM_SET}/"
