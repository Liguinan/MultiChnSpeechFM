#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os

import numpy as np
import torch


def main():
    
    print(f"args.snapshots:{args.snapshots}")

    num = len(args.snapshots)

    avg = None

    # sum
    for path in args.snapshots:
        # import pdb; pdb.set_trace()
        states = torch.load(path, map_location=torch.device("cpu"))
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]

    # average
    for k in avg.keys():
        if avg[k] is not None:
            if avg[k].is_floating_point():
                avg[k] /= num
            else:
                avg[k] //= num

    torch.save(avg, args.out)


def get_parser():
    parser = argparse.ArgumentParser(description="average models from snapshot")
    parser.add_argument("--snapshots", required=True, type=str, nargs="+")
    parser.add_argument("--out", required=True, type=str)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main()
