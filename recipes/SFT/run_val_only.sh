#!/usr/bin/env bash
# Smoke-test validation only (no training).
#
# Fresh init then 1 val pass:
#   gpu_ids=0 bash run_val_only.sh
#
# Resume latest ckpt then 1 val pass:
#   FULLUTT_RESUME=1 gpu_ids=0 bash run_val_only.sh

set -eo pipefail
cd "$(dirname "$0")"
unset RESUME

gpu_ids="${gpu_ids:-0}"
CFG="${CFG:-TAC_Based_MultiChnNet/config/train_moe_01346_fullutt.toml}"
FULLUTT_RESUME="${FULLUTT_RESUME:-0}"

BACKBONE_CKPT="${BACKBONE_CKPT:-../../../MultiChnSpeechFMExperiments_DOA/v2/TAC_Based_MultiChnNet_doa_phase2_fullutt/train_doa_phase2_fullutt/checkpoints/best_model.tar}"
MOE_ADAPTER_CKPT="${MOE_ADAPTER_CKPT:-/path/to/MultiChnSpeechFMExperiments/v6_moe_sisnr_kl_inner_adapter_ce_all_params_pretraining/TAC_Based_MultiChnNet_train_batch_mode_moe_adapters_position_0_fixed_array_01346/allhours_kl_0.01/train_batch_mode_moe_adapter_ce_allhours_kl_weight_0.01/checkpoints/best_model.tar}"

CKPT_DIR="${CKPT_DIR:-../../../MultiChnSpeechFMExperiments_DOA/v5_moe_from_v2_doa_backbone_fullutt/TAC_Based_MultiChnNet_doa_moe_01346_fullutt/train_moe_01346_fullutt/checkpoints}"
LATEST_CKPT="${CKPT_DIR}/latest_model.tar"

export BACKBONE_CKPT MOE_ADAPTER_CKPT
num_processes=$(echo "$gpu_ids" | tr "," "\n" | wc -l | tr -d ' ')

if [[ "${FULLUTT_RESUME}" == "1" ]]; then
  if [[ ! -f "${LATEST_CKPT}" ]]; then
    echo "ERROR: FULLUTT_RESUME=1 but missing ${LATEST_CKPT}"
    exit 1
  fi
  TRAIN_ARGS=(-R -V)
  echo "VAL-ONLY resume <- ${LATEST_CKPT}"
else
  if [[ ! -f "${BACKBONE_CKPT}" ]]; then
    echo "ERROR: BACKBONE_CKPT (v2) not found: ${BACKBONE_CKPT}"
    exit 1
  fi
  if [[ ! -f "${MOE_ADAPTER_CKPT}" ]]; then
    echo "ERROR: MOE_ADAPTER_CKPT not found: ${MOE_ADAPTER_CKPT}"
    exit 1
  fi
  TRAIN_ARGS=(-P "${BACKBONE_CKPT}" -V)
  echo "VAL-ONLY fresh: v2=${BACKBONE_CKPT}"
  echo "               moe_adapters=${MOE_ADAPTER_CKPT}"
fi

echo "CFG=${CFG}  gpu_ids=${gpu_ids}"
echo "python args: ${TRAIN_ARGS[*]}"

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${gpu_ids} torchrun \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  --nnodes=1 --nproc_per_node="${num_processes}" \
  train_moe_fullutt.py \
  -C "${CFG}" \
  "${TRAIN_ARGS[@]}"
