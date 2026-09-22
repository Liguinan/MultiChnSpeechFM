#!/usr/bin/env bash
# Convenience launcher for DOA-finetune ablations.
#
#   DOA_FINETUNE_MODE=full|head|head_conv gpu_ids=0 bash run.sh
#   DOA_FINETUNE_MODE=head FULLUTT_RESUME=1 gpu_ids=0 bash run.sh

set -eo pipefail
cd "$(dirname "$0")"

export gpu_ids="${gpu_ids:-0}"
export BACKBONE_CKPT="${BACKBONE_CKPT:-/path/to/MultiChnSpeechFMExperiments_DOA/pretraining/TAC_Based_MultiChnNet_doa_phase2/train_doa_phase2/checkpoints/model_0012_best.pth}"
export MOE_ADAPTER_CKPT="${MOE_ADAPTER_CKPT:-/path/to/MultiChnSpeechFMExperiments/v6_moe_sisnr_kl_inner_adapter_ce_all_params_pretraining/TAC_Based_MultiChnNet_train_batch_mode_moe_adapters_position_0_fixed_array_01346/allhours_kl_0.01/train_batch_mode_moe_adapter_ce_allhours_kl_weight_0.01/checkpoints/best_model.tar}"
export FULLUTT_RESUME="${FULLUTT_RESUME:-0}"
export DOA_FINETUNE_MODE="${DOA_FINETUNE_MODE:-head}"

echo ">>> run.sh -> run_moe_fullutt.sh"
echo "    gpu_ids=${gpu_ids}"
echo "    DOA_FINETUNE_MODE=${DOA_FINETUNE_MODE}"
echo "    BACKBONE_CKPT=${BACKBONE_CKPT}"
echo "    MOE_ADAPTER_CKPT=${MOE_ADAPTER_CKPT}"
echo "    FULLUTT_RESUME=${FULLUTT_RESUME}"

bash run_moe_fullutt.sh
