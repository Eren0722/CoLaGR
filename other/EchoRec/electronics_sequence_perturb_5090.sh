#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "${ROOT_DIR}/experiments/electronics_sequence_perturb_5090.sh" "$@"
