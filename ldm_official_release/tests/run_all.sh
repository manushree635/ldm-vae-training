#!/bin/bash
# Run the CPU test suite. No GPU, no checkpoints, no data; stdlib unittest only.
#
#   bash tests/run_all.sh            # everything, ~3 min
#   bash tests/run_all.sh --fast     # skip the end-to-end model steps, ~30 s
#   PYTHON=venv_ldm3/bin/python bash tests/run_all.sh
#
# The repo root is wherever this file lives, so a copy of tests/ dropped into
# ldm_official_release/ checks that tree instead.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "${HERE}")"

if [[ -z "${PYTHON:-}" ]]; then
    for cand in "${ROOT}/venv_ldm/bin/python" "${ROOT}/venv/bin/python" "$(command -v python)"; do
        if [[ -x "${cand}" ]]; then PYTHON="${cand}"; break; fi
    done
fi
[[ "${PYTHON}" = /* ]] || PYTHON="$(cd "$(dirname "${PYTHON}")" && pwd)/$(basename "${PYTHON}")"

if [[ "${1:-}" == "--fast" ]]; then export LDM_SKIP_SLOW=1; fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"          # login-node manners
export TORCH_HOME="${TORCH_HOME:-/projectnb/ivc-ml/mvasu6/.cache/torch}"

echo "python: ${PYTHON}"
echo "root  : ${ROOT}"
cd "${HERE}"
exec "${PYTHON}" -W ignore -m unittest -v test_static test_configs test_modules test_model
