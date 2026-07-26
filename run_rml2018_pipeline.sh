#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CONFIG="${CONFIG:-configs/rml2018_tgfm_v7_curriculum.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs}"
BRANCH_MODE="${BRANCH_MODE:-time_only}"
EXPERT_SEED="${EXPERT_SEED:-123}"
RUN_TAG="${RUN_TAG:-time_only_seed${EXPERT_SEED}}"

latest_run() {
  local prefix="$1"
  find "${OUTPUT_ROOT}" -maxdepth 1 -type d -name "${prefix}_*" -printf '%T@ %p\n' \
    | sort -n | tail -1 | cut -d' ' -f2-
}

if [[ -z "${TEACHER_RUN:-}" ]]; then
  python scripts/train_teacher.py --config "${CONFIG}"
  TEACHER_RUN="$(latest_run teacher)"
fi

if [[ -z "${ROUTER_RUN:-}" ]]; then
  python scripts/train_reliability_router.py \
    --config "${CONFIG}" \
    --teacher-run "${TEACHER_RUN}"
  ROUTER_RUN="$(latest_run v7_reliability_router)"
fi

python scripts/train_raw_low_snr_expert.py \
  --config "${CONFIG}" \
  --teacher-run "${TEACHER_RUN}" \
  --router-run "${ROUTER_RUN}" \
  --branch-mode "${BRANCH_MODE}" \
  --seed-override "${EXPERT_SEED}" \
  --run-tag "${RUN_TAG}"

EXPERT_RUN="$(latest_run "v8_raw_low_snr_expert_${RUN_TAG}")"

python scripts/evaluate_v8_raw_expert_diagnostics.py \
  --config "${CONFIG}" \
  --teacher-run "${TEACHER_RUN}" \
  --router-run "${ROUTER_RUN}" \
  --expert-run "${EXPERT_RUN}" \
  --output-tag "${RUN_TAG}"

printf 'TGFM_PIPELINE_OK\nteacher=%s\nrouter=%s\nexpert=%s\n' \
  "${TEACHER_RUN}" "${ROUTER_RUN}" "${EXPERT_RUN}"
