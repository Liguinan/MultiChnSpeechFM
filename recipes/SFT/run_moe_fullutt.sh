#!/usr/bin/env bash
# ModelDOAMoE debug: MoE + optional DOA subset finetune.
# Modes: full | head | head_conv | none

set -eo pipefail
cd "$(dirname "$0")"
unset RESUME

gpu_ids="${gpu_ids:-0}"
FULLUTT_RESUME="${FULLUTT_RESUME:-0}"

# If user only sets CFG, let toml control doa_finetune_mode (do not force env default).
USER_CFG="${CFG:-}"
if [[ -n "${USER_CFG}" ]]; then
  CFG="${USER_CFG}"
  # Keep DOA_FINETUNE_MODE only if user explicitly set it; else unset so toml wins.
  if [[ -z "${DOA_FINETUNE_MODE+x}" ]]; then
    :
  elif [[ -z "${DOA_FINETUNE_MODE}" ]]; then
    unset DOA_FINETUNE_MODE
  fi
  # For resume path lookup when mode env not set, parse from CFG name if possible.
  if [[ -z "${DOA_FINETUNE_MODE:-}" ]]; then
    case "${CFG}" in
      *doa_full*) DOA_FINETUNE_MODE=full ;;
      *doa_head_conv*) DOA_FINETUNE_MODE=head_conv ;;
      *doa_head*) DOA_FINETUNE_MODE=head ;;
      *) DOA_FINETUNE_MODE=none ;;
    esac
  fi
else
  DOA_FINETUNE_MODE="${DOA_FINETUNE_MODE:-head}"
  case "${DOA_FINETUNE_MODE}" in
    full|head|head_conv|none) ;;
    *)
      echo "ERROR: DOA_FINETUNE_MODE must be full|head|head_conv|none, got: ${DOA_FINETUNE_MODE}"
      exit 1
      ;;
  esac
  if [[ "${DOA_FINETUNE_MODE}" == "none" ]]; then
    CFG="TAC_Based_MultiChnNet/config/train_moe_01346_fullutt.toml"
  else
    CFG="TAC_Based_MultiChnNet/config/train_moe_01346_doa_${DOA_FINETUNE_MODE}.toml"
  fi
  export DOA_FINETUNE_MODE
fi

BACKBONE_CKPT="${BACKBONE_CKPT:-}"
MOE_ADAPTER_CKPT="${MOE_ADAPTER_CKPT:-}"

if [[ "${DOA_FINETUNE_MODE}" == "none" ]]; then
  CKPT_DIR="${CKPT_DIR:-../../../MultiChnSpeechFMExperiments_DOA/v5_moe_from_v2_doa_backbone_debug/TAC_Based_MultiChnNet_doa_moe_01346/train_moe_01346_fullutt/checkpoints}"
else
  CKPT_DIR="${CKPT_DIR:-../../../MultiChnSpeechFMExperiments_DOA/v5_moe_from_v2_doa_backbone_debug/TAC_Based_MultiChnNet_doa_moe_01346_${DOA_FINETUNE_MODE}/train_moe_01346_doa_${DOA_FINETUNE_MODE}/checkpoints}"
fi
LATEST_CKPT="${CKPT_DIR}/latest_model.tar"

export BACKBONE_CKPT MOE_ADAPTER_CKPT
num_processes=$(echo "$gpu_ids" | tr "," "\n" | wc -l | tr -d ' ')

if [[ "${FULLUTT_RESUME}" == "1" ]]; then
  if [[ ! -f "${LATEST_CKPT}" ]]; then
    echo "ERROR: FULLUTT_RESUME=1 but missing ${LATEST_CKPT}"
    exit 1
  fi
  TRAIN_ARGS=(-R)
  echo "FULLUTT_RESUME=1 <- ${LATEST_CKPT}"
else
  TRAIN_ARGS=()
  if [[ -n "${BACKBONE_CKPT}" ]]; then
    if [[ ! -f "${BACKBONE_CKPT}" ]]; then
      echo "ERROR: BACKBONE_CKPT not found: ${BACKBONE_CKPT}"
      exit 1
    fi
    TRAIN_ARGS+=(-P "${BACKBONE_CKPT}")
    echo "Fresh: backbone=${BACKBONE_CKPT}"
  else
    echo "Fresh: backbone from toml [init].backbone_path"
  fi
  if [[ -n "${MOE_ADAPTER_CKPT}" ]]; then
    if [[ ! -f "${MOE_ADAPTER_CKPT}" ]]; then
      echo "ERROR: MOE_ADAPTER_CKPT not found: ${MOE_ADAPTER_CKPT}"
      exit 1
    fi
    echo "       moe_adapters=${MOE_ADAPTER_CKPT}"
  else
    echo "       moe_adapters from toml [init].moe_adapter_ckpt"
  fi
fi

echo "CFG=${CFG}  DOA_FINETUNE_MODE=${DOA_FINETUNE_MODE:-"(from toml)"}  gpu_ids=${gpu_ids}"
echo "python args: ${TRAIN_ARGS[*]:-(none)}"

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${gpu_ids} torchrun \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  --nnodes=1 --nproc_per_node="${num_processes}" \
  train_moe_fullutt.py \
  -C "${CFG}" \
  "${TRAIN_ARGS[@]}"
