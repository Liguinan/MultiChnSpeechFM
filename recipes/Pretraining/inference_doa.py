#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Inference entry for ModelDOA (phase2)."""
import argparse
import os
import sys
from pathlib import Path

import toml

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))
from audio_zen.utils import initialize_module


def main(config, checkpoint_path, output_dir):
    inferencer_class = initialize_module(config["inferencer"]["path"], initialize=False)
    inferencer = inferencer_class(config, checkpoint_path, output_dir)
    inferencer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference DOA (Scheme v2)")
    parser.add_argument("-C", "--configuration", type=str, required=True)
    parser.add_argument("-M", "--model_checkpoint_path", type=str, required=True)
    parser.add_argument("-O", "--output_dir", type=str, required=True)
    args = parser.parse_args()

    config_path = Path(args.configuration).expanduser().absolute()
    configuration = toml.load(config_path.as_posix())

    # recipe root + TAC_Based_MultiChnNet (for audio_feature_doa, model_doa, ...)
    recipe_root = Path(__file__).expanduser().absolute().parent
    sys.path.append(recipe_root.as_posix())
    sys.path.append((recipe_root / "TAC_Based_MultiChnNet").as_posix())

    main(configuration, args.model_checkpoint_path, args.output_dir)
