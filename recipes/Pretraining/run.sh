#!/usr/bin/env bash
# Convenience launcher for DOA pretraining.
#   PHASE=1|2|all gpu_ids=0 bash run.sh

set -eo pipefail
cd "$(dirname "$0")"
echo ">>> run.sh -> run_doa.sh (PHASE=${PHASE:-1})"
bash run_doa.sh
