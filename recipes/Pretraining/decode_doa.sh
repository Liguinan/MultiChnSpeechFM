#!/usr/bin/env bash
# Inference for pretraining ModelDOA (phase2 checkpoint)
#
#   CKPT=/path/to/phase2/best_model.tar \
#   OUT=/path/to/inference_out \
#   GPU=0 \
#   bash decode_doa.sh

set -euo pipefail
cd "$(dirname "$0")"

GPU=${GPU:-0}
CKPT=${CKPT:-"../../../MultiChnSpeechFMExperiments_DOA/pretraining/TAC_Based_MultiChnNet_doa_phase2/train_doa_phase2/checkpoints/best_model.tar"}
OUT=${OUT:-"../../../MultiChnSpeechFMExperiments_DOA/pretraining/TAC_Based_MultiChnNet_doa_phase2/inference_test_clean"}
CFG=${CFG:-"TAC_Based_MultiChnNet/config/inference_doa.toml"}

if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT}"
  exit 1
fi

echo "GPU=${GPU}"
echo "CKPT=${CKPT}"
echo "OUT=${OUT}"
echo "CFG=${CFG}"

CUDA_VISIBLE_DEVICES=${GPU} python inference_doa.py \
  -C "${CFG}" \
  -M "${CKPT}" \
  -O "${OUT}"
