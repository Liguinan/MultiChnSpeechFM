# ModelDOAMoE debug: DOA finetune ablation (+ MoE adapter/router)

Copied from `multichn_librispeech_v5_moe_from_v2_doa_backbone_fullutt`.

## Ablation switch

In config `[init]`:

```toml
freeze_backbone = true
doa_finetune_mode = "full"   # or "head" | "head_conv" | "none"
```

| Mode | Trainable (sep backbone frozen) |
|------|----------------------------------|
| `full` | entire DOA module + MoE adapter + router |
| `head` | `doa_head` + MoE adapter + router |
| `head_conv` | `doa_head` + `conv1x1_pre` + MoE adapter + router |
| `none` | MoE adapter + router only |

DOA full module = `ln_LPS_doa`, `conv1x1_pre`, `audio_block_*_doa`, `tac*_doa`, `doa_head`.

Env override: `DOA_FINETUNE_MODE` > toml `init.doa_finetune_mode`.

## Run

```bash
cd recipes/multichn_librispeech_v5_moe_from_v2_doa_backbone_debug

# (1) full DOA + MoE
DOA_FINETUNE_MODE=full gpu_ids=0 bash run.sh

# (2) doa_head + MoE
DOA_FINETUNE_MODE=head gpu_ids=0 bash run.sh

# (3) doa_head + conv1x1_pre + MoE
DOA_FINETUNE_MODE=head_conv gpu_ids=0 bash run.sh

# resume
DOA_FINETUNE_MODE=head FULLUTT_RESUME=1 gpu_ids=0 bash run.sh
```

Optional ckpt overrides: `BACKBONE_CKPT=... MOE_ADAPTER_CKPT=...`.

## Outputs

`MultiChnSpeechFMExperiments_DOA/v5_moe_from_v2_doa_backbone_debug/TAC_Based_MultiChnNet_doa_moe_01346_{full,head,head_conv}/`
