#!/usr/bin/env bash
# DOA pretraining: phase1 DOA-only, phase2 merge DOA + sep.
# Each phase: warmup head-4s (10 ep, bs=36) → fullutt (50 ep, bs=8, accum=4).
#
#   PHASE=1 gpu_ids=0 bash run_doa.sh
#   PHASE=2 gpu_ids=0 PHASE1_CKPT=... PRETRAIN_CKPT=... bash run_doa.sh
#   PHASE=all gpu_ids=0 bash run_doa.sh
#
# Resume current phase (loads checkpoints/.../latest_model.tar):
#   PHASE=1 RESUME=1 gpu_ids=0 bash run_doa.sh
#   PHASE=2 RESUME=1 gpu_ids=0 bash run_doa.sh

set -euo pipefail
cd "$(dirname "$0")"

gpu_ids=${gpu_ids:-0}
PRETRAIN_CKPT=${PRETRAIN_CKPT:-"/path/to/MultiChnSpeechFMExperiments/v2/TAC_Based_MultiChnNet_first_channel_for_masking/train/checkpoints/best_model.tar"}
PHASE1_CKPT=${PHASE1_CKPT:-"../../../MultiChnSpeechFMExperiments_DOA/pretraining/TAC_Based_MultiChnNet_doa_phase1/train_doa_phase1/checkpoints/best_model.tar"}
PHASE=${PHASE:-"1"}
RESUME=${RESUME:-0}

num_processes=$(echo "$gpu_ids" | tr "," "\n" | wc -l | tr -d ' ')

run_phase1() {
  echo "======== Pretraining Phase 1 (DOA-only, warmup→fullutt) ========"
  EXTRA=()
  if [[ "${RESUME}" == "1" ]]; then
    EXTRA+=(-R)
    echo "RESUME=1"
  fi
  OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${gpu_ids} torchrun \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    --nnodes=1 --nproc_per_node="${num_processes}" \
    train_doa.py \
    -C TAC_Based_MultiChnNet/config/train_doa_phase1.toml \
    "${EXTRA[@]}"
}

run_phase2() {
  echo "======== Pretraining Phase 2 (merge DOA + sep, warmup→fullutt) ========"
  EXTRA=()
  if [[ "${RESUME}" == "1" ]]; then
    EXTRA+=(-R)
    echo "RESUME=1"
  else
    if [[ ! -f "${PHASE1_CKPT}" ]]; then
      echo "ERROR: PHASE1_CKPT not found: ${PHASE1_CKPT}"
      exit 1
    fi
    if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
      echo "ERROR: PRETRAIN_CKPT not found: ${PRETRAIN_CKPT}"
      exit 1
    fi
    EXTRA+=(-P "${PHASE1_CKPT}")
    export PRETRAIN_CKPT
    echo "PHASE1_CKPT=${PHASE1_CKPT}"
    echo "PRETRAIN_CKPT=${PRETRAIN_CKPT}"
  fi
  OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${gpu_ids} torchrun \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    --nnodes=1 --nproc_per_node="${num_processes}" \
    train_doa.py \
    -C TAC_Based_MultiChnNet/config/train_doa_phase2.toml \
    "${EXTRA[@]}"
}

case "${PHASE}" in
  1) run_phase1 ;;
  2) run_phase2 ;;
  all)
    RESUME=0
    run_phase1
    wait
    run_phase2
    ;;
  *)
    echo "Unknown PHASE=${PHASE}, use 1|2|all"
    exit 1
    ;;
esac
