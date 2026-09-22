#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MoE full-utterance training: backbone from v2 DOA + 4 adapters."""
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
    if warmup_cfg.get("enabled", False):
        warmup_args = dict(config["train_dataset"]["args"])
        warmup_args["training_utt_fixed_length"] = int(
            warmup_cfg.get("training_utt_fixed_length", 4)
        )
        warmup_args["crop_mode"] = warmup_cfg.get("crop_mode", "head")
        train_dataset = initialize_module(
            "dataset.Dataset", args=warmup_args, initialize=True
        )
    validation_dataset = initialize_module(
        config["validation_dataset"]["path"], args=config["validation_dataset"]["args"]
    )

    train_batch_size = config["train_dataset"]["dataloader"]["batch_size"]
    if warmup_cfg.get("enabled", False):
        train_batch_size = int(warmup_cfg.get("batch_size", train_batch_size))
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
    optimizer = torch.optim.Adam(
        params=model.parameters(),
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
    if only_validation:
        # trainer already destroyed PG; force a clean process exit for torchrun
        import sys
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoE from v2 DOA backbone (fullutt)")
    parser.add_argument("-C", "--configuration", required=True, type=str)
    parser.add_argument("-R", "--resume", action="store_true")
    parser.add_argument("-V", "--only_validation", action="store_true")
    parser.add_argument(
        "-P",
        "--preloaded_model_path",
        type=str,
        default=None,
        help="Optional override for init.backbone_path (v2 ModelDOA phase2 ckpt).",
    )
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    if args.preloaded_model_path and args.resume:
        raise RuntimeError("Use either -R or -P, not both.")

    configuration = toml.load(Path(args.configuration).expanduser().absolute().as_posix())
    recipe_root = Path(__file__).expanduser().absolute().parent
    sys.path.append(recipe_root.as_posix())
    sys.path.append((recipe_root / "TAC_Based_MultiChnNet").as_posix())

    if args.resume:
        configuration.setdefault("meta", {})["preloaded_model_path"] = ""
    else:
        # Backbone: CLI -P > env BACKBONE_CKPT > init.backbone_path
        backbone = (
            args.preloaded_model_path
            or os.environ.get("BACKBONE_CKPT")
            or configuration.get("init", {}).get("backbone_path")
        )
        if backbone:
            configuration.setdefault("init", {})["backbone_path"] = backbone
            configuration["meta"]["preloaded_model_path"] = backbone
        else:
            raise RuntimeError(
                "Need backbone: set init.backbone_path, BACKBONE_CKPT, or -P."
            )

        env_moe = os.environ.get("MOE_ADAPTER_CKPT")
        if env_moe:
            configuration.setdefault("init", {})["moe_adapter_ckpt"] = env_moe

    # DOA finetune mode: env DOA_FINETUNE_MODE > init.doa_finetune_mode
    env_mode = os.environ.get("DOA_FINETUNE_MODE")
    if env_mode:
        configuration.setdefault("init", {})["doa_finetune_mode"] = env_mode.strip().lower()
    doa_mode = str(
        configuration.get("init", {}).get("doa_finetune_mode", "none")
    ).strip().lower()
    configuration.setdefault("init", {})["doa_finetune_mode"] = doa_mode

    cfg_name = Path(args.configuration).name
    experiment_name, _ = os.path.splitext(cfg_name)
    configuration["meta"]["experiment_name"] = experiment_name
    configuration["meta"]["config_path"] = args.configuration

    if local_rank == 0:
        print(f"[train] resume={args.resume}")
        print(f"[train] v2 backbone={configuration['meta'].get('preloaded_model_path')}")
        print(f"[train] moe_adapter_ckpt={configuration.get('init', {}).get('moe_adapter_ckpt')}")
        print(
            f"[train] freeze_backbone={configuration.get('init', {}).get('freeze_backbone')} "
            f"doa_finetune_mode={doa_mode} save_dir={configuration['meta'].get('save_dir')}"
        )
        warmup = configuration.get("warmup", {})
        if warmup.get("enabled", False):
            w_ep = int(warmup.get("epochs", 10))
            f_ep = int(configuration["trainer"]["train"]["epochs"])
            print(
                f"[train] warmup={w_ep} epochs (head {warmup.get('training_utt_fixed_length', 4)}s, "
                f"bs={warmup.get('batch_size', 36)}, accum={warmup.get('grad_accum_steps', 1)}), "
                f"then fullutt={f_ep} epochs "
                f"(bs={configuration['train_dataset']['dataloader']['batch_size']}, "
                f"accum={configuration['trainer']['train']['grad_accum_steps']}), "
                f"total={w_ep + f_ep}"
            )

    entry(local_rank, configuration, args.resume, args.only_validation)
