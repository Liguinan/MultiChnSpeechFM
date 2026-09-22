# multichn_librispeech_pretraining

DOA pretraining recipe based on **v2 ModelDOA**, with **v5-style warmup→fullutt** data loading.

## What it does

| Phase | Model | Loss | Train data | Val |
|-------|-------|------|------------|-----|
| **1** | DOA-only | DOA CE | warmup 10 ep head-4s → fullutt 50 ep | fullutt |
| **2** | DOA + sep (pred-AF) | SI-SNR + 0.1·DOA CE | same schedule | fullutt |

- **Init phase2:** merge phase1 DOA + GT-DOA pretrained sep (v2 way)
- **Arrays:** multi-array rotation like v2 (`[0,2,3,5]`, `[0,2,4,6]`, `[0,1,2,3,4,5]`, `[0,1,2,3,4,5,6]`)
- **Data:** original `multi_chn_simu` (not rx1.5); keep v2 `mic_position`
- **Warmup crop:** from 0s, one 4s segment per utt; zero-pad if shorter
- **Hyperparams:** warmup bs=36 accum=1; fullutt bs=8 accum=4

## Run

```bash
cd recipes/multichn_librispeech_pretraining

# Phase1
PHASE=1 gpu_ids=0 bash run.sh

# Phase2 (after phase1)
PHASE=2 gpu_ids=0 \
  PHASE1_CKPT=../../../MultiChnSpeechFMExperiments_DOA/pretraining/TAC_Based_MultiChnNet_doa_phase1/train_doa_phase1/checkpoints/best_model.tar \
  PRETRAIN_CKPT=/path/to/GT-DOA-sep/best_model.tar \
  bash run.sh

# Or both
PHASE=all gpu_ids=0 bash run.sh

# Resume (loads that phase's latest_model.tar; use RESUME=1, not RESUME=2)
PHASE=1 RESUME=1 gpu_ids=0 bash run.sh
PHASE=2 RESUME=1 gpu_ids=0 bash run.sh
```

## Inference

```bash
CKPT=../../../MultiChnSpeechFMExperiments_DOA/pretraining/TAC_Based_MultiChnNet_doa_phase2/train_doa_phase2/checkpoints/best_model.tar \
OUT=../../../MultiChnSpeechFMExperiments_DOA/pretraining/TAC_Based_MultiChnNet_doa_phase2/inference_test_clean \
GPU=0 \
bash decode_doa.sh
```

## Outputs

- Phase1: `MultiChnSpeechFMExperiments_DOA/pretraining/TAC_Based_MultiChnNet_doa_phase1/`
- Phase2: `MultiChnSpeechFMExperiments_DOA/pretraining/TAC_Based_MultiChnNet_doa_phase2/`

Total epochs per phase = 10 (warmup) + 50 (fullutt) = **60**.
