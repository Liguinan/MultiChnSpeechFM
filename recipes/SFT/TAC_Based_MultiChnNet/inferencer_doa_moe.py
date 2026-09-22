#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Inferencer for ModelDOAMoE: pred-DOA AF + MoE separation."""
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

import audio_zen.acoustics.result_analysis as ra
from audio_zen.inferencer.base_inferencer import BaseInferencer
from model_doa_moe import angle_rad_to_bin, bin_to_angle_rad


class InferencerDOAMoE(BaseInferencer):
    def __init__(self, config, checkpoint_path, output_dir):
        super().__init__(config, checkpoint_path, output_dir)
        doa_cfg = config.get("doa", {})
        model_args = config["model"]["args"]
        self.num_doa_bins = int(
            doa_cfg.get("num_doa_bins", model_args.get("num_doa_bins", 12))
        )
        self.doa_angle_range_deg = float(
            doa_cfg.get(
                "doa_angle_range_deg", model_args.get("doa_angle_range_deg", 180.0)
            )
        )

    @staticmethod
    def _load_model(model_config, checkpoint_path, device):
        from audio_zen.utils import initialize_module

        model = initialize_module(
            model_config["path"], args=model_config["args"], initialize=True
        )
        model_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.suffix == ".pth":
            model_static_dict = model_checkpoint
            parts = checkpoint_path.stem.split("_")
            epoch = parts[1] if len(parts) > 1 else "0"
        else:
            model_static_dict = model_checkpoint["model"]
            epoch = model_checkpoint.get("epoch", 0)
        model_static_dict = {
            k.replace("module.", ""): v for k, v in model_static_dict.items()
        }
        print(f"Loading ModelDOAMoE checkpoint (epoch == {epoch})...")
        missing, unexpected = model.load_state_dict(model_static_dict, strict=False)
        print(f"  missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print(f"  missing (first 8): {missing[:8]}")
        model.to(device)
        model.eval()
        if hasattr(model, "use_pred_doa_for_af"):
            model.use_pred_doa_for_af = True
        return model, epoch

    @torch.no_grad()
    def __call__(self):
        sisnr_lst, pesq_lst, stoi_lst = [], [], []
        doa_correct, doa_total = 0, 0
        exception_utt_lst = []
        angle_err_deg_lst = []

        for i, data in tqdm(enumerate(self.dataloader), desc="Inference-DOA-MoE"):
            assert len(data["mix"]) == 1, "Inference batch size must be 1."
            data["geometry"] = torch.from_numpy(np.array(self.geometry))
            data["geometry_id"] = torch.tensor(0, dtype=torch.long)
            data = self.offload_data(data, self.device)
            data["mix"] = data["mix"] / 32768
            data["ref"] = data["ref"] / 32768

            ests, _adapter_outputs, _geometry_logits, doa_logits = self.model(
                [
                    data["mix"],
                    data["src_doa"],
                    data["spk_num"],
                    data["ori_wav_len"],
                    data["geometry"],
                    data["geometry_id"],
                ]
            )

            tgt_angle = data["src_doa"][:, 0]
            tgt_bin = angle_rad_to_bin(
                tgt_angle, self.num_doa_bins, self.doa_angle_range_deg
            )
            pred_bin = doa_logits.argmax(dim=-1)
            doa_correct += int((pred_bin == tgt_bin).sum().item())
            doa_total += int(tgt_bin.numel())

            pred_angle = bin_to_angle_rad(
                pred_bin, self.num_doa_bins, self.doa_angle_range_deg
            )
            err_rad = (pred_angle - tgt_angle).abs()
            err_deg = (err_rad * 180.0 / math.pi).item()
            angle_err_deg_lst.append(err_deg)

            noisy = data["mix"].detach().squeeze(0).cpu().numpy()[0, :]
            clean = data["ref"].detach().squeeze(0).squeeze(0).cpu().numpy()
            enhanced = ests[0].detach().squeeze(0).cpu().numpy()

            norm = np.linalg.norm(noisy, np.inf)
            enhanced = enhanced * norm / max(np.max(np.abs(enhanced)), 1e-8)

            min_len = min(len(clean), len(enhanced))
            clean = clean[:min_len]
            enhanced = enhanced[:min_len]

            try:
                sisnr = ra.get_SI_SNR(enhanced, clean)
                pesq = ra.get_PESQ(enhanced, clean)
                stoi = ra.get_STOI(enhanced, clean)
            except Exception:
                print(
                    f"Exception utterance: idx-{i}, wav_name:{data['wav_file_lst'][0]}"
                )
                exception_utt_lst.append(data["wav_file_lst"][0])
                continue

            sisnr_lst.append(sisnr)
            pesq_lst.append(pesq)
            stoi_lst.append(stoi)

            pred_deg = float(pred_angle.item() * 180.0 / math.pi)
            tgt_deg = float(tgt_angle.item() * 180.0 / math.pi)
            line = (
                f"idx:{i}, wav:{data['wav_file_lst'][0]}, "
                f"sisnr:{sisnr:.3f}, pesq:{pesq:.3f}, stoi:{stoi:.3f}, "
                f"doa_pred:{pred_deg:.1f}, doa_gt:{tgt_deg:.1f}, "
                f"doa_bin:{int(pred_bin.item())}/{int(tgt_bin.item())}, "
                f"err_deg:{err_deg:.1f}"
            )
            print(line)
            with open(self.log_dir / "log.txt", "a") as fw:
                fw.write(line + "\n")

            sf.write(
                self.enhanced_dir / f"{data['wav_file_lst'][0]}.wav",
                enhanced,
                samplerate=self.acoustic_config["sr"],
            )

        n = max(len(sisnr_lst), 1)
        avg_sisnr = sum(sisnr_lst) / n
        avg_pesq = sum(pesq_lst) / n
        avg_stoi = sum(stoi_lst) / n
        doa_acc = doa_correct / max(doa_total, 1)
        avg_err = sum(angle_err_deg_lst) / max(len(angle_err_deg_lst), 1)
        summary = (
            f"geometry:{self.geometry}, avg_sisnr:{avg_sisnr:.4f}, "
            f"avg_pesq:{avg_pesq:.4f}, avg_stoi:{avg_stoi:.4f}, "
            f"doa_acc:{doa_acc:.4f}, avg_angle_err_deg:{avg_err:.2f}"
        )
        print(summary)
        with open(self.log_dir / "log.txt", "a") as fw:
            fw.write(summary + "\n")
        print(f"all exception utterances: {exception_utt_lst}")


if __name__ == "__main__":
    pass
