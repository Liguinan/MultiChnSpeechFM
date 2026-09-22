import matplotlib.pyplot as plt
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
import numpy as np

from pathlib import Path

from audio_zen.acoustics.feature import drop_band
from audio_zen.acoustics.mask import build_complex_ideal_ratio_mask, decompress_cIRM
from audio_zen.trainer.base_trainer_moe_all_pretrainig_all_params import BaseTrainer
from itertools import permutations
from torch.nn import CrossEntropyLoss

from init_from_v2_and_adapters import (
    load_state_dict,
    map_v2_doa_to_doa_moe,
    extract_expert_adapters,
    merge_doa_moe_init,
)
from model_doa_moe import angle_rad_to_bin

plt.switch_backend("agg")


class Trainer(BaseTrainer):
    def __init__(
        self,
        dist,
        rank,
        config,
        resume,
        only_validation,
        model,
        loss_function,
        optimizer,
        train_dataloader,
        validation_dataloader,
        training_array_geometries,
        sampler
    ):
        # Read init options before super() because BaseTrainer may call _preload_model.
        self.config = config
        self.init_cfg = config.get("init", {})
        self.freeze_backbone = bool(self.init_cfg.get("freeze_backbone", False))
        # DOA finetune ablation (with freeze_backbone=true):
        #   none       - MoE adapter + router only
        #   full       - entire DOA module + MoE
        #   head       - doa_head + MoE
        #   head_conv  - doa_head + conv1x1_pre + MoE
        self.doa_finetune_mode = str(
            self.init_cfg.get("doa_finetune_mode", "none")
        ).strip().lower()
        doa_cfg = config.get("doa", {})
        self.lambda_doa = float(doa_cfg.get("lambda_doa", 0.1))
        self.num_doa_bins = int(
            doa_cfg.get(
                "num_doa_bins",
                config["model"]["args"].get("num_doa_bins", 12),
            )
        )
        self.doa_angle_range_deg = float(
            doa_cfg.get(
                "doa_angle_range_deg",
                config["model"]["args"].get("doa_angle_range_deg", 180.0),
            )
        )
        # Freeze + rebuild Adam BEFORE BaseTrainer resume/preload so optimizer
        # param-group size matches checkpoints saved after the same rebuild.
        if self.freeze_backbone:
            self._apply_partial_trainable(model, self.doa_finetune_mode)
            lr = float(config["optimizer"]["lr"])
            wd = float(config["optimizer"]["lr_decay"])
            optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad],
                lr=lr,
                weight_decay=wd,
            )
        super().__init__(
            dist, rank, config, resume, only_validation, model, loss_function, optimizer
        )
        self.fullutt_grad_accum_steps = max(
            1, int(self.train_config.get("grad_accum_steps", 1))
        )
        self.grad_accum_steps = self.fullutt_grad_accum_steps
        self.train_dataloader = train_dataloader
        self.valid_dataloader = validation_dataloader
        self.training_array_geometries = training_array_geometries
        self.sampler = sampler
        self.num_spks = 1
        self.ada_num = config['model']['args']['ada_num']
        self.kl_loss_weight = config["trainer"]["train"]["kl_loss_weight"]
        self.kl_loss = config["trainer"]["train"]["kl_loss"]
        self.ce_loss_weight = config["trainer"]["train"]["ce_loss_weight"]
        self.ce_loss = config["model"]["args"]["ce_loss"]

        warmup_cfg = config.get("warmup", {})
        self.warmup_enabled = bool(warmup_cfg.get("enabled", False))
        self.warmup_epochs = int(warmup_cfg.get("epochs", 0))
        self.warmup_batch_size = int(warmup_cfg.get("batch_size", 36))
        self.warmup_grad_accum_steps = max(1, int(warmup_cfg.get("grad_accum_steps", 1)))
        self.fullutt_batch_size = int(config["train_dataset"]["dataloader"]["batch_size"])
        self.fullutt_epochs = int(config["trainer"]["train"]["epochs"])
        if self.warmup_enabled:
            self.epochs = self.warmup_epochs + self.fullutt_epochs
        self._current_train_phase = None
        self._maybe_switch_train_dataloader(self.start_epoch)
        self._set_trainable_scope()

    def _rebuild_train_dataloader(self, use_fullutt):
        from torch.utils.data import DataLoader

        from dataset import collate_fn, DistributedStratifiedSampler
        from audio_zen.utils import initialize_module

        args = dict(self.config["train_dataset"]["args"])
        dl_cfg = dict(self.config["train_dataset"]["dataloader"])
        if use_fullutt:
            dataset_path = "dataset_fullutt.Dataset"
            args["training_utt_fixed_length"] = -1
            batch_size = self.fullutt_batch_size
            self.grad_accum_steps = self.fullutt_grad_accum_steps
        else:
            dataset_path = "dataset.Dataset"
            warmup_cfg = self.config.get("warmup", {})
            args["training_utt_fixed_length"] = int(
                warmup_cfg.get("training_utt_fixed_length", 4)
            )
            args["crop_mode"] = warmup_cfg.get("crop_mode", "head")
            batch_size = self.warmup_batch_size
            self.grad_accum_steps = self.warmup_grad_accum_steps

        train_dataset = initialize_module(dataset_path, args=args, initialize=True)
        self.sampler = DistributedStratifiedSampler(
            dataset=train_dataset,
            rank=self.rank,
            shuffle=True,
            seed=self.config["meta"]["seed"],
            batch_size=batch_size,
            num_strata=dl_cfg["num_strata"],
        )
        self.train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_sampler=self.sampler,
            collate_fn=collate_fn,
            num_workers=dl_cfg["num_workers"],
            pin_memory=dl_cfg["pin_memory"],
        )

    def _maybe_switch_train_dataloader(self, epoch):
        if not self.warmup_enabled:
            return
        use_fullutt = epoch > self.warmup_epochs
        phase = "fullutt" if use_fullutt else "warmup_head_crop"
        if phase == self._current_train_phase:
            return
        self._rebuild_train_dataloader(use_fullutt)
        self._current_train_phase = phase
        self.dist.barrier()
        if self.rank == 0:
            if use_fullutt:
                print(
                    f"[Trainer] Warmup finished ({self.warmup_epochs} epochs). "
                    f"Switching to full-utterance: batch_size={self.fullutt_batch_size}, "
                    f"grad_accum_steps={self.grad_accum_steps} (epoch {epoch})."
                )
            else:
                print(
                    f"[Trainer] Warmup: first {self.warmup_epochs} epochs use "
                    f"head {self.config['warmup'].get('training_utt_fixed_length', 4)}s crop, "
                    f"batch_size={self.warmup_batch_size}, "
                    f"grad_accum_steps={self.grad_accum_steps}."
                )

    def _preload_model(self, model_path):
        """
        ModelDOAMoE init:
          - all non-adapter params <- v2 ModelDOA phase2 (backbone_path / -P)
          - expert_adapter_1..6    <- MoE ckpt (moe_adapter_ckpt)
          - moe_weights stay random unless init.load_moe_weights=true
        """
        backbone_path = self.init_cfg.get("backbone_path") or model_path
        moe_ckpt = self.init_cfg.get("moe_adapter_ckpt")
        if not moe_ckpt:
            raise RuntimeError(
                "init.moe_adapter_ckpt is required (original MoE v5/v6 best_model.tar)."
            )

        backbone_path = Path(backbone_path).expanduser().absolute()
        moe_ckpt = Path(moe_ckpt).expanduser().absolute()
        assert backbone_path.exists(), f"v2 backbone not found: {backbone_path}"
        assert moe_ckpt.exists(), f"MoE adapter ckpt not found: {moe_ckpt}"

        v2_mapped = map_v2_doa_to_doa_moe(
            load_state_dict(backbone_path),
            copy_fusion_to_all=bool(self.init_cfg.get("copy_fusion_to_all", False)),
        )
        moe_adapters = extract_expert_adapters(
            load_state_dict(moe_ckpt),
            load_moe_weights=bool(self.init_cfg.get("load_moe_weights", False)),
        )

        raw = (
            self.model.module
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
            else self.model
        )
        merged = merge_doa_moe_init(
            model_state_keys=list(raw.state_dict().keys()),
            v2_mapped=v2_mapped,
            moe_adapter_state=moe_adapters,
        )
        missing, unexpected = raw.load_state_dict(merged, strict=False)
        if self.rank == 0:
            print(
                f"[Trainer] ModelDOAMoE init: v2={backbone_path}; "
                f"moe_adapters={moe_ckpt}; "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )

    # DOA module param-name prefixes in ModelDOAMoE.
    _DOA_FULL_PREFIXES = (
        "ln_LPS_doa",
        "conv1x1_pre",
        "audio_block_1_doa",
        "tac1_doa",
        "audio_block_2_doa",
        "tac2_doa",
        "doa_head",
    )

    @classmethod
    def _is_doa_trainable_param(cls, name, mode):
        """Return whether a DOA param should be trainable under doa_finetune_mode."""
        if mode in ("", "none", "moe_only"):
            return False
        if mode == "full":
            return any(
                name == p or name.startswith(p + ".") for p in cls._DOA_FULL_PREFIXES
            )
        if mode == "head":
            return name == "doa_head" or name.startswith("doa_head.")
        if mode in ("head_conv", "head+conv", "conv_head"):
            return (
                name == "doa_head"
                or name.startswith("doa_head.")
                or name == "conv1x1_pre"
                or name.startswith("conv1x1_pre.")
            )
        raise ValueError(
            f"Unknown init.doa_finetune_mode={mode!r}. "
            "Use one of: none | full | head | head_conv"
        )

    @classmethod
    def _apply_partial_trainable(cls, model, doa_finetune_mode):
        """Freeze all except MoE adapter/router (+ optional DOA subset)."""
        unfrozen = []
        for name, param in model.named_parameters():
            is_moe = ("expert_adapter" in name) or ("moe_weights" in name)
            is_doa = cls._is_doa_trainable_param(name, doa_finetune_mode)
            if is_moe or is_doa:
                param.requires_grad = True
                unfrozen.append(name)
            else:
                param.requires_grad = False
        return unfrozen

    def _set_trainable_scope(self):
        raw = (
            self.model.module
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
            else self.model
        )
        if not self.freeze_backbone:
            total_params, trainable_params, _ = self._count_parameters(raw)
            if self.rank == 0:
                print(
                    f"trainable_params: {trainable_params}, total_params: {total_params}, "
                    f"ratio: {trainable_params / max(total_params, 1):.4f} (full finetune)"
                )
            return

        mode = self.doa_finetune_mode
        # Re-apply freeze on DDP-wrapped module (idempotent; optimizer already rebuilt).
        unfrozen = self._apply_partial_trainable(raw, mode)
        total_params, trainable_params, _ = self._count_parameters(raw)
        if self.rank == 0:
            print(
                f"[Trainer] Partial finetune: doa_finetune_mode={mode!r}, "
                f"unfrozen {len(unfrozen)} param tensors "
                f"(MoE adapter/router + selected DOA)"
            )
            for name in unfrozen[:16]:
                print(f"  trainable: {name}")
            if len(unfrozen) > 16:
                print(f"  ... and {len(unfrozen) - 16} more")
            print(
                f"trainable_params: {trainable_params}, total_params: {total_params}, "
                f"ratio: {trainable_params / max(total_params, 1):.4f}"
            )
            print(
                f"[Trainer] optimizer param tensors: "
                f"{sum(len(g['params']) for g in self.optimizer.param_groups)}"
            )

    def _doa_loss(self, doa_logits, src_doa):
        tgt_bin = angle_rad_to_bin(
            src_doa[:, 0], self.num_doa_bins, self.doa_angle_range_deg
        )
        return torch.nn.functional.cross_entropy(doa_logits, tgt_bin)

    def _doa_acc(self, doa_logits, src_doa):
        tgt_bin = angle_rad_to_bin(
            src_doa[:, 0], self.num_doa_bins, self.doa_angle_range_deg
        )
        return (doa_logits.argmax(dim=-1) == tgt_bin).float().mean()

    def offload_data(self, data, device):
        def _cuda(obj):
            return obj.to(device) if isinstance(obj, torch.Tensor) else obj

        def offload(obj):
            obj_cuda = _cuda(obj)
            if isinstance(obj_cuda, list):
                obj_cuda = list(map(_cuda, obj_cuda))
            return obj_cuda

        new_data = dict()
        for key in data.keys():
            new_data[key] = offload(data[key])
        return new_data

    def sisnr(self, x, s, eps=1e-8):
        """
        Arguments:
        x: separated signal, BS x S
        s: reference signal, BS x S
        Return:
        sisnr: BS tensor
        """

        def l2norm(mat, keepdim=False):
            return torch.norm(mat, dim=-1, keepdim=keepdim)

        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape != s.shape:
            raise RuntimeError(
                "Dimension mismatch when calculate si-snr, {} vs {}".format(
                    x.shape, s.shape))
        x_zm = x - torch.mean(x, dim=-1, keepdim=True)
        s_zm = s - torch.mean(s, dim=-1, keepdim=True)
        t = torch.sum(
            x_zm * s_zm, dim=-1,
            keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True) ** 2 + eps)
        return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

    def get_array_geometry(self, epoch, stratum_id, array_lst):
        """
        根据epoch和stratum_id获取对应的方法

        Args:
            epoch: 当前epoch（从1开始）
            stratum_id: stratum的ID（0,1,2,3）

        Returns:
            对应的处理方法
        """
        num_array = len(array_lst)
        epoch = epoch - 1

        # 计算偏移量：每个epoch偏移1位
        shift = epoch % num_array

        # 计算当前stratum应该使用哪个方法
        # (stratum_id - shift) % num_methods 确保循环
        array_idx = (stratum_id - shift) % num_array

        selected_array = array_lst[array_idx]

        return selected_array

    def _train_epoch(self, epoch, save_meta_file_path=None):
        # save_meta_file_path kept for BaseTrainer API compatibility; unused.
        self._maybe_switch_train_dataloader(epoch)
        if self.warmup_enabled and self.rank == 0:
            phase = "warmup" if epoch <= self.warmup_epochs else "fullutt"
            print(f"[Trainer] epoch {epoch} train phase: {phase}")
        loss_total = 0.0

        self.sampler.set_epoch(epoch)
        self.optimizer.zero_grad(set_to_none=True)
        num_batches = len(self.train_dataloader)
        for batch_idx, data in (
            enumerate(tqdm(self.train_dataloader, desc="Training"))
            if self.rank == 0
            else enumerate(self.train_dataloader)
        ):
            assert len(self.training_array_geometries) >= 1, self.training_array_geometries
            if len(self.training_array_geometries) == 1:
                geometry = self.training_array_geometries[0]
                stratum_id = 0
            else:
                stratum_id = batch_idx % 4
                stratum_id_new, _ = self.sampler.get_stratum_info(
                    batch_idx + self.sampler.start_batch
                )
                assert stratum_id == stratum_id_new, (
                    f"stratum_id=batch_idx % 4:{stratum_id}, "
                    f"stratum_id from sampler:{stratum_id_new}"
                )
                geometry = self.get_array_geometry(
                    epoch, stratum_id, self.training_array_geometries
                )
            print(f"selected geometry: {geometry}, expert_id: {stratum_id}")

            data["geometry"] = torch.from_numpy(np.array(geometry))
            data["geometry_id"] = torch.tensor(stratum_id, dtype=torch.long)
            data = self.offload_data(data, self.rank)
            data["mix"] = data["mix"] / 32768
            data["ref"] = data["ref"] / 32768

            with autocast(enabled=self.use_amp):
                ests, adapter_outputs, geometry_logits, doa_logits = self.model(
                    [
                        data["mix"],
                        data["src_doa"],
                        data["spk_num"],
                        data["ori_wav_len"],
                        data["geometry"],
                        data["geometry_id"],
                    ]
                )
                refs = data["ref"]

                def sisnr_loss(permute):
                    return sum(
                        [self.sisnr(ests[s], refs[:, t]) for s, t in enumerate(permute)]
                    ) / len(permute)

                sisnr_mat = torch.stack(
                    [sisnr_loss(p) for p in permutations(range(self.num_spks))]
                )
                max_perutt, _ = torch.max(sisnr_mat, dim=0)
                loss_sisnr = -torch.sum(max_perutt) / data["mix"].size(0)
                loss = loss_sisnr
                print(f"sisnr speech loss: {loss_sisnr.item():.4f}")

                loss_doa = self._doa_loss(doa_logits, data["src_doa"])
                doa_acc = self._doa_acc(doa_logits, data["src_doa"])
                loss = loss + self.lambda_doa * loss_doa
                print(f"doa ce: {loss_doa.item():.4f}, doa_acc: {doa_acc.item():.4f}")

                if self.ce_loss:
                    ce_loss = 0.0
                    loss_fct = CrossEntropyLoss()
                    for geometry_logit in geometry_logits:
                        ce_loss += loss_fct(
                            geometry_logit,
                            data["geometry_id"].expand(geometry_logit.shape[0]),
                        )
                    loss = loss - self.ce_loss_weight * ce_loss
                    print(f"ce loss: {ce_loss}")

                if self.kl_loss:
                    KL_loss = 0.0
                    KL_criterion = torch.nn.KLDivLoss(reduction="mean", log_target=True)
                    for adapter_output in adapter_outputs:
                        log_probs = [
                            torch.nn.functional.log_softmax(hs, dim=-1)
                            for hs in torch.unbind(adapter_output, dim=0)
                        ]
                        for i in range(self.ada_num):
                            for j in range(i + 1, self.ada_num):
                                kl_i_j = KL_criterion(log_probs[i], log_probs[j].detach())
                                kl_j_i = KL_criterion(log_probs[j], log_probs[i].detach())
                                KL_loss += (kl_i_j + kl_j_i) / 2.0
                    loss = loss - self.kl_loss_weight * KL_loss
                    print(f"kl loss: {-KL_loss}")
                print(f"all loss: {loss}")

            scaled_loss = loss / self.grad_accum_steps
            self.scaler.scale(scaled_loss).backward()

            is_accum_step = (batch_idx + 1) % self.grad_accum_steps == 0
            is_last_batch = (batch_idx + 1) == num_batches
            if is_accum_step or is_last_batch:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip_grad_norm_value
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            loss_total += loss.item()

        if self.rank == 0:
            self.writer.add_scalar(
                "Loss/Train", loss_total / len(self.train_dataloader), epoch
            )

    @torch.no_grad()
    def _validation_epoch(self, epoch, save_val_loss_path):
        visualization_n_samples = self.visualization_config["n_samples"]

        loss_total = 0.0
        sisnr_total = 0.0
        doa_acc_total = 0.0
        item_idx_cnt = 0
        geometry_cnt = len(self.training_array_geometries)

        for stratum_id, geometry in enumerate(self.training_array_geometries):
            for i, data in tqdm(
                enumerate(self.valid_dataloader), desc=f"Validation...{geometry}"
            ):
                assert len(data["mix"]) == 1, "The batch size for the validation stage must be one."
                data["geometry"] = torch.from_numpy(np.array(geometry))
                data["geometry_id"] = torch.tensor(stratum_id, dtype=torch.long)
                data = self.offload_data(data, self.rank)
                data["mix"] = data["mix"] / 32768
                data["ref"] = data["ref"] / 32768

                ests, adapter_outputs, geometry_logits, doa_logits = self.model(
                    [
                        data["mix"],
                        data["src_doa"],
                        data["spk_num"],
                        data["ori_wav_len"],
                        data["geometry"],
                        data["geometry_id"],
                    ]
                )
                refs = data["ref"]

                def sisnr_loss(permute):
                    return sum(
                        [self.sisnr(ests[s], refs[:, t]) for s, t in enumerate(permute)]
                    ) / len(permute)

                sisnr_mat = torch.stack(
                    [sisnr_loss(p) for p in permutations(range(self.num_spks))]
                )
                max_perutt, _ = torch.max(sisnr_mat, dim=0)
                # max_perutt is SI-SNR (dB); training minimizes -SI-SNR
                sisnr = torch.sum(max_perutt) / data["mix"].size(0)
                loss = -sisnr
                doa_acc = self._doa_acc(doa_logits, data["src_doa"])
                print(
                    f"sisnr: {sisnr.item():.4f}, sisnr_loss: {loss.item():.4f}, "
                    f"doa_acc: {doa_acc.item():.4f}"
                )

                noisy = data["mix"].detach().squeeze(0).cpu().numpy()[0, :]
                clean = data["ref"].detach().squeeze(0).squeeze(0).cpu().numpy()
                enhanced = ests[0].detach().squeeze(0).cpu().numpy()
                assert len(noisy) == len(clean) == len(enhanced)

                loss_total += loss.item()
                sisnr_total += sisnr.item()
                doa_acc_total += doa_acc.item()
                item_idx_cnt += 1

                if item_idx_cnt <= visualization_n_samples:
                    self.spec_audio_visualization(
                        noisy, enhanced, clean, data["wav_file_lst"][0], epoch
                    )

        n = max(geometry_cnt * len(self.valid_dataloader), 1)
        avg_loss_total = loss_total / n
        avg_sisnr = sisnr_total / n
        avg_doa_acc = doa_acc_total / n

        self.writer.add_scalar("Loss/Validation_SI_SNR_loss", avg_loss_total, epoch)
        self.writer.add_scalar("Metric/Validation_SI_SNR", avg_sisnr, epoch)
        self.writer.add_scalar("Metric/Validation_DOA_acc", avg_doa_acc, epoch)

        line = (
            f"epoch {epoch}: sisnr {avg_sisnr:.4f}, "
            f"sisnr_loss {avg_loss_total:.4f}, doa_acc {avg_doa_acc:.4f}\n"
        )
        if save_val_loss_path:
            with open(save_val_loss_path, "a") as fw:
                fw.write(line)
        print(f"[val] {line.strip()}")
        return avg_loss_total

    def train(self):
        """Override: only-validation runs once and writes val_loss.txt."""
        if not self.only_validation:
            return super().train()

        # All ranks must participate in barriers so torchrun exits cleanly.
        self.dist.barrier()
        if self.rank == 0:
            epoch = self.start_epoch
            print(f"{'=' * 15} only-validation (epoch {epoch}) {'=' * 15}")
            self._set_models_to_eval_mode()
            self.val_loss_dir.mkdir(parents=True, exist_ok=True)
            val_loss_file = (self.val_loss_dir / "val_loss.txt").as_posix()
            metric_score = self._validation_epoch(
                epoch, save_val_loss_path=val_loss_file
            )
            print(f"[only-validation] done. score={metric_score}")
        self.dist.barrier()
        self.dist.destroy_process_group()
        return
