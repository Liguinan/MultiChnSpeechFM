#!/usr/bin/env bash
# Inference for ModelDOAMoE (pred-DOA AF + MoE sep) on test / val / train.
#
# Usage:
#   CKPT=/path/to/best_model.tar \
#   OUT=/path/to/inference_out \
#   GPU=0 \
#   bash decode_doa_moe.sh

set -euo pipefail
cd "$(dirname "$0")"

GPU=${GPU:-0}
CKPT=${CKPT:-"/path/to/MultiChnSpeechFMExperiments_DOA/v5_moe_from_v2_doa_backbone_debug/TAC_Based_MultiChnNet_doa_moe_01346_full/100hours/lambda_doa=0.6/train_moe_01346_doa_full/checkpoints/model_0017.pth"}
OUT=${OUT:-"/path/to/MultiChnSpeechFMExperiments_DOA/v5_moe_from_v2_doa_backbone_debug/TAC_Based_MultiChnNet_doa_moe_01346_full/100hours/lambda_doa=0.6/inference_moe_01346_doa_full"}

CFG_TEST=${CFG_TEST:-"TAC_Based_MultiChnNet/config/inference_doa_moe.toml"}
CFG_VAL=${CFG_VAL:-"TAC_Based_MultiChnNet/config/inference_doa_moe_dev.toml"}
CFG_TRAIN=${CFG_TRAIN:-"TAC_Based_MultiChnNet/config/inference_doa_moe_train.toml"}

if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT}"
  exit 1
fi

echo "GPU=${GPU}"
echo "CKPT=${CKPT}"
echo "OUT=${OUT}"

run_one() {
  local cfg="$1"
  local tag="$2"
  echo "======== decode ${tag}: ${cfg} ========"
  CUDA_VISIBLE_DEVICES=${GPU} python inference_doa_moe.py \
    -C "${cfg}" \
    -M "${CKPT}" \
    -O "${OUT}"
}

run_one "${CFG_TEST}" "test"
run_one "${CFG_VAL}" "val"
run_one "${CFG_TRAIN}" "train"
