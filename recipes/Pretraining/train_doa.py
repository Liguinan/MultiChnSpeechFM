#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Training entry for DOA pretraining (warmup→fullutt, phase1/2)."""
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import toml
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))
from dataset import collate_fn, DistributedStratifiedSampler
import audio_zen.loss as loss
from audio_zen.utils import initialize_module


def entry(rank, config, resume, only_validation):
    torch.manual_seed(config["meta"]["seed"])
    np.random.seed(config["meta"]["seed"])
    random.seed(config["meta"]["seed"])
    torch.cuda.set_device(rank)

    torch.distributed.init_process_group(backend="nccl")
    print(f"Process {rank + 1} initialized.")

    train_dataset = initialize_module(
        config["train_dataset"]["path"], args=config["train_dataset"]["args"]
    )
    warmup_cfg = config.get("warmup", {})
    train_batch_size = config["train_dataset"]["dataloader"]["batch_size"]
    if warmup_cfg.get("enabled", False):
        warmup_args = dict(config["train_dataset"]["args"])
        warmup_args["training_utt_fixed_length"] = int(
            warmup_cfg.get("training_utt_fixed_length", 4)
        )
        warmup_args["crop_mode"] = warmup_cfg.get("crop_mode", "head")
        train_dataset = initialize_module(
            "dataset.Dataset", args=warmup_args, initialize=True
        )
        train_batch_size = int(warmup_cfg.get("batch_size", train_batch_size))

    validation_dataset = initialize_module(
        config["validation_dataset"]["path"], args=config["validation_dataset"]["args"]
    )

    sampler = DistributedStratifiedSampler(
        dataset=train_dataset,
        rank=rank,
        shuffle=True,
        seed=config["meta"]["seed"],
        batch_size=train_batch_size,
        num_strata=config["train_dataset"]["dataloader"]["num_strata"],
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=config["train_dataset"]["dataloader"]["num_workers"],
        pin_memory=config["train_dataset"]["dataloader"]["pin_memory"],
    )

    valid_dataloader = DataLoader(
        dataset=validation_dataset,
        collate_fn=collate_fn,
        **config["validation_dataset"]["dataloader"],
    )

    model = initialize_module(config["model"]["path"], args=config["model"]["args"])

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters. Check training_phase / freeze logic.")
    optimizer = torch.optim.Adam(
        params=trainable_params,
        lr=config["optimizer"]["lr"],
        weight_decay=(config["optimizer"]["lr_decay"]),
    )

    loss_function = getattr(loss, config["loss_function"]["name"])(
        **config["loss_function"]["args"]
    )

    trainer_class = initialize_module(config["trainer"]["path"], initialize=False)
    trainer = trainer_class(
        dist=dist,
        rank=rank,
        config=config,
        resume=resume,
        only_validation=only_validation,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        validation_dataloader=valid_dataloader,
        training_array_geometries=config["array_geometry"]["training_array_geometries"],
        sampler=sampler,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MultiChanSpeechFM DOA pretraining")
    parser.add_argument(
        "-C", "--configuration", required=True, type=str, help="Configuration (*.toml)."
    )
    parser.add_argument(
        "-R",
        "--resume",
        action="store_true",
        help="Resume the experiment from latest checkpoint.",
    )
    parser.add_argument(
        "-V",
        "--only_validation",
        action="store_true",
        help="Only run validation, which is used for debugging.",
    )
    parser.add_argument(
        "-P",
        "--preloaded_model_path",
        type=str,
        default=None,
        help="Phase2: path to phase1 best_model.tar. Phase1: omit.",
    )
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])

    if args.preloaded_model_path:
        assert not args.resume, "The 'resume' conflicts with the 'preloaded_model_path'."

    config_path = Path(args.configuration).expanduser().absolute()
    configuration = toml.load(config_path.as_posix())

    recipe_root = Path(__file__).expanduser().absolute().parent
    sys.path.append(recipe_root.as_posix())
    sys.path.append((recipe_root / "TAC_Based_MultiChnNet").as_posix())

    env_sep = os.environ.get("PRETRAIN_CKPT")
    if env_sep:
        configuration["meta"]["sep_pretrained_path"] = env_sep

    cfg_name = Path(args.configuration).name
    if "phase2" in cfg_name:
        experiment_name = "train_doa_phase2"
    elif "phase1" in cfg_name:
        experiment_name = "train_doa_phase1"
    else:
        experiment_name, _ = os.path.splitext(cfg_name)

    configuration["meta"]["experiment_name"] = experiment_name
    configuration["meta"]["config_path"] = args.configuration
    if args.resume:
        configuration["meta"]["preloaded_model_path"] = ""
    else:
        configuration["meta"]["preloaded_model_path"] = args.preloaded_model_path

    if local_rank == 0:
        warmup = configuration.get("warmup", {})
        if warmup.get("enabled", False):
            w_ep = int(warmup.get("epochs", 10))
            f_ep = int(configuration["trainer"]["train"]["epochs"])
            print(
                f"[train] warmup={w_ep} epochs (head {warmup.get('training_utt_fixed_length', 4)}s, "
                f"bs={warmup.get('batch_size', 36)}, accum={warmup.get('grad_accum_steps', 1)}), "
                f"then fullutt={f_ep} epochs "
                f"(bs={configuration['train_dataset']['dataloader']['batch_size']}, "
                f"accum={configuration['trainer']['train'].get('grad_accum_steps', 1)}), "
                f"total={w_ep + f_ep}"
            )

    entry(local_rank, configuration, args.resume, args.only_validation)
