#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Staged trainer for ModelDOA (pretraining recipe with warmup→fullutt)."""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm
from itertools import permutations
from pathlib import Path

from audio_zen.trainer.base_trainer import BaseTrainer
from model_doa import angle_rad_to_bin

plt.switch_backend("agg")


class TrainerDOA(BaseTrainer):
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
        sampler,
    ):
        # BaseTrainer never stores self.config; read options before super()
        # because BaseTrainer may call _preload_model().
        self.config = config
        doa_cfg = config.get("doa", {})
        self.training_phase = int(doa_cfg.get("training_phase", 1))
        self.lambda_doa = float(doa_cfg.get("lambda_doa", 0.1))
        self.num_doa_bins = int(doa_cfg.get("num_doa_bins", 18))
        self.doa_angle_range_deg = float(doa_cfg.get("doa_angle_range_deg", 180.0))
        self.sep_pretrained_path = config["meta"].get("sep_pretrained_path", None)

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

        warmup_cfg = config.get("warmup", {})
        self.warmup_enabled = bool(warmup_cfg.get("enabled", False))
        self.warmup_epochs = int(warmup_cfg.get("epochs", 0))
        self.warmup_batch_size = int(warmup_cfg.get("batch_size", 36))
        self.warmup_grad_accum_steps = max(
            1, int(warmup_cfg.get("grad_accum_steps", 1))
        )
        self.fullutt_batch_size = int(config["train_dataset"]["dataloader"]["batch_size"])
        self.fullutt_epochs = int(config["trainer"]["train"]["epochs"])
        if self.warmup_enabled:
            self.epochs = self.warmup_epochs + self.fullutt_epochs
        self._current_train_phase = None
        self._maybe_switch_train_dataloader(self.start_epoch)

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
                    f"[TrainerDOA] Warmup finished ({self.warmup_epochs} epochs). "
                    f"Switching to full-utterance: batch_size={self.fullutt_batch_size}, "
                    f"grad_accum_steps={self.grad_accum_steps} (epoch {epoch})."
                )
            else:
                print(
                    f"[TrainerDOA] Warmup: first {self.warmup_epochs} epochs use "
                    f"head {self.config['warmup'].get('training_utt_fixed_length', 4)}s crop, "
                    f"batch_size={self.warmup_batch_size}, "
                    f"grad_accum_steps={self.grad_accum_steps}."
                )

    def _raw_model(self):
        return (
            self.model.module
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
            else self.model
        )

    def _preload_model(self, model_path):
        """
        Phase2: model_path = phase1 DOA ckpt; also load sep from
        meta.sep_pretrained_path and merge.
        Phase1 should not call this (no -P).
        """
        model_path = model_path.expanduser().absolute()
        assert model_path.exists(), f"The file {model_path.as_posix()} does not exist."
        raw_model = self._raw_model()

        if self.training_phase == 2:
            if not self.sep_pretrained_path:
                raise RuntimeError(
                    "Phase2 requires meta.sep_pretrained_path (GT-DOA pretrained sep)."
                )
            sep_path = Path(self.sep_pretrained_path).expanduser().absolute()
            assert sep_path.exists(), f"sep_pretrained_path not found: {sep_path}"
            raw_model.training_phase = self.training_phase
            raw_model.merge_phase1_doa_and_sep_pretrained(
                model_path.as_posix(), sep_path.as_posix(), map_location="cpu"
            )
            trainable = [p for p in raw_model.parameters() if p.requires_grad]
            if not trainable:
                raise RuntimeError("Phase2 merge left no trainable parameters.")
            lr = self.optimizer.param_groups[0]["lr"]
            wd = self.optimizer.param_groups[0].get("weight_decay", 0.0)
            self.optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=wd)
            if self.rank == 0:
                print(
                    f"[TrainerDOA] rebuilt optimizer after merge: "
                    f"{sum(p.numel() for p in trainable)} trainable params"
                )
        else:
            checkpoint = torch.load(model_path.as_posix(), map_location="cpu")
            state = checkpoint["model"] if "model" in checkpoint else checkpoint
            state = {k.replace("module.", ""): v for k, v in state.items()}
            missing, unexpected = raw_model.load_state_dict(state, strict=False)
            raw_model.training_phase = self.training_phase
            raw_model._apply_training_phase()
            if self.rank == 0:
                print(
                    f"[TrainerDOA] loaded ckpt from {model_path}; "
                    f"missing={len(missing)}, unexpected={len(unexpected)}"
                )

        if self.rank == 0:
            n_train = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in raw_model.parameters())
            print(f"[TrainerDOA] trainable params: {n_train}/{n_total}")

    def offload_data(self, data, device):
        def _cuda(obj):
            return obj.to(device) if isinstance(obj, torch.Tensor) else obj

        def offload(obj):
            obj_cuda = _cuda(obj)
            if isinstance(obj_cuda, list):
                obj_cuda = list(map(_cuda, obj_cuda))
            return obj_cuda

        return {key: offload(data[key]) for key in data.keys()}

    def sisnr(self, x, s, eps=1e-8):
        def l2norm(mat, keepdim=False):
            return torch.norm(mat, dim=-1, keepdim=keepdim)

        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape != s.shape:
            raise RuntimeError(
                "Dimension mismatch when calculate si-snr, {} vs {}".format(
                    x.shape, s.shape
                )
            )
        x_zm = x - torch.mean(x, dim=-1, keepdim=True)
        s_zm = s - torch.mean(s, dim=-1, keepdim=True)
        t = torch.sum(x_zm * s_zm, dim=-1, keepdim=True) * s_zm / (
            l2norm(s_zm, keepdim=True) ** 2 + eps
        )
        return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

    def get_array_geometry(self, epoch, stratum_id, array_lst):
        num_array = len(array_lst)
        epoch = epoch - 1
        shift = epoch % num_array
        array_idx = (stratum_id - shift) % num_array
        return array_lst[array_idx]

    def _doa_loss(self, doa_logits, src_doa):
        """Target speaker DOA (index 0) -> CE."""
        tgt_angle = src_doa[:, 0]
        tgt_bin = angle_rad_to_bin(
            tgt_angle, self.num_doa_bins, self.doa_angle_range_deg
        )
        return F.cross_entropy(doa_logits, tgt_bin)

    def _doa_acc(self, doa_logits, src_doa):
        tgt_bin = angle_rad_to_bin(
            src_doa[:, 0], self.num_doa_bins, self.doa_angle_range_deg
        )
        pred_bin = doa_logits.argmax(dim=-1)
        return (pred_bin == tgt_bin).float().mean()

    def _sisnr_loss(self, ests, refs, batch_size):
        def sisnr_loss(permute):
            return sum(
                [self.sisnr(ests[s], refs[:, t]) for s, t in enumerate(permute)]
            ) / len(permute)

        sisnr_mat = torch.stack(
            [sisnr_loss(p) for p in permutations(range(self.num_spks))]
        )
        max_perutt, _ = torch.max(sisnr_mat, dim=0)
        return -torch.sum(max_perutt) / batch_size

    def _prepare_batch(self, data, epoch, batch_idx):
        assert len(self.training_array_geometries) >= 1
        if len(self.training_array_geometries) == 1:
            geometry = self.training_array_geometries[0]
        else:
            stratum_id = batch_idx % 4
            stratum_id_new, _ = self.sampler.get_stratum_info(
                batch_idx + self.sampler.start_batch
            )
            assert stratum_id == stratum_id_new
            geometry = self.get_array_geometry(
                epoch, stratum_id, self.training_array_geometries
            )
        data["geometry"] = torch.from_numpy(np.array(geometry))
        data = self.offload_data(data, self.rank)
        data["mix"] = data["mix"] / 32768
        data["ref"] = data["ref"] / 32768
        return data, geometry

    def _train_epoch(self, epoch, save_meta_file_path=None):
        self._maybe_switch_train_dataloader(epoch)
        if self.warmup_enabled and self.rank == 0:
            phase = "warmup" if epoch <= self.warmup_epochs else "fullutt"
            print(f"[TrainerDOA] epoch {epoch} train phase: {phase}")

        loss_total = 0.0
        doa_acc_total = 0.0
        self.sampler.set_epoch(epoch)

        iterator = (
            enumerate(tqdm(self.train_dataloader, desc="Training"))
            if self.rank == 0
            else enumerate(self.train_dataloader)
        )

        self.optimizer.zero_grad(set_to_none=True)
        num_batches = len(self.train_dataloader)

        for batch_idx, data in iterator:
            data, geometry = self._prepare_batch(data, epoch, batch_idx)
            if self.rank == 0:
                print(f"selected geometry: {geometry}")

            model_input = [
                data["mix"],
                data["src_doa"],
                data["spk_num"],
                data["ori_wav_len"],
                data["geometry"],
            ]

            with autocast(enabled=self.use_amp):
                ests, doa_logits = self.model(model_input)
                loss_doa = self._doa_loss(doa_logits, data["src_doa"])
                doa_acc = self._doa_acc(doa_logits, data["src_doa"])

                if self.training_phase == 1:
                    loss = loss_doa
                else:
                    loss_sisnr = self._sisnr_loss(
                        ests, data["ref"], data["mix"].size(0)
                    )
                    loss = loss_sisnr + self.lambda_doa * loss_doa

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
            doa_acc_total += doa_acc.item()
            if self.rank == 0:
                print(f"loss: {loss.item():.4f}, doa_acc: {doa_acc.item():.4f}")

        if self.rank == 0:
            n = len(self.train_dataloader)
            self.writer.add_scalar("Loss/Train", loss_total / n, epoch)
            self.writer.add_scalar("DOA/Train_Acc", doa_acc_total / n, epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch, save_val_loss_path):
        visualization_n_samples = self.visualization_config["n_samples"]

        loss_total = 0.0
        sisnr_total = 0.0
        doa_acc_total = 0.0
        item_idx_cnt = 0
        geometry_cnt = len(self.training_array_geometries)

        for geometry in self.training_array_geometries:
            for i, data in tqdm(
                enumerate(self.valid_dataloader), desc=f"Validation...{geometry}"
            ):
                assert len(data["mix"]) == 1, "Validation batch size must be 1."
                data["geometry"] = torch.from_numpy(np.array(geometry))
                data = self.offload_data(data, self.rank)
                data["mix"] = data["mix"] / 32768
                data["ref"] = data["ref"] / 32768

                ests, doa_logits = self.model(
                    [
                        data["mix"],
                        data["src_doa"],
                        data["spk_num"],
                        data["ori_wav_len"],
                        data["geometry"],
                    ]
                )

                loss_doa = self._doa_loss(doa_logits, data["src_doa"])
                doa_acc = self._doa_acc(doa_logits, data["src_doa"])

                if self.training_phase == 1:
                    loss = loss_doa
                else:
                    loss_sisnr = self._sisnr_loss(
                        ests, data["ref"], data["mix"].size(0)
                    )
                    loss = loss_sisnr + self.lambda_doa * loss_doa
                    sisnr_total += (-loss_sisnr).item()

                    noisy = data["mix"].detach().squeeze(0).cpu().numpy()[0, :]
                    clean = data["ref"].detach().squeeze(0).squeeze(0).cpu().numpy()
                    enhanced = ests[0].detach().squeeze(0).cpu().numpy()
                    assert len(noisy) == len(clean) == len(enhanced)
                    item_idx_cnt += 1
                    if item_idx_cnt <= visualization_n_samples:
                        self.spec_audio_visualization(
                            noisy, enhanced, clean, data["wav_file_lst"][0], epoch
                        )

                loss_total += loss.item()
                doa_acc_total += doa_acc.item()

        n = geometry_cnt * len(self.valid_dataloader)
        avg_loss = loss_total / n
        avg_doa_acc = doa_acc_total / n
        avg_sisnr = sisnr_total / n if self.training_phase != 1 else float("nan")

        self.writer.add_scalar("Loss/Validation_Total", avg_loss, epoch)
        self.writer.add_scalar("DOA/Validation_Acc", avg_doa_acc, epoch)
        if self.training_phase != 1:
            self.writer.add_scalar("SI_SNR/Validation", avg_sisnr, epoch)

        if save_val_loss_path:
            with open(save_val_loss_path, "a") as fw:
                if self.training_phase == 1:
                    fw.write(
                        "epoch {}: loss {:.4f}, doa_acc {:.4f}\n".format(
                            epoch, avg_loss, avg_doa_acc
                        )
                    )
                else:
                    fw.write(
                        "epoch {}: loss {:.4f}, sisnr {:.4f}, doa_acc {:.4f}\n".format(
                            epoch, avg_loss, avg_sisnr, avg_doa_acc
                        )
                    )
        if self.training_phase == 1:
            print(f"val loss: {avg_loss:.4f}, doa_acc: {avg_doa_acc:.4f}")
        else:
            print(
                f"val loss: {avg_loss:.4f}, sisnr: {avg_sisnr:.4f}, "
                f"doa_acc: {avg_doa_acc:.4f}"
            )

        if self.training_phase == 1:
            return -avg_doa_acc
        return avg_loss
