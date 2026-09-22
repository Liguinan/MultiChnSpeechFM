#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/12/5 11:12
# @Author  : Guinan Li
# @File    : stat_array_geometry_distribution_in_training.py
from tqdm import tqdm

data_dir = "/path/to/MultiChnSpeechFMExperiments/v2/TAC_Based_MultiChnNet_first_channel_for_masking/train/meta_files"
file_name_lst = ["epoch_3.txt", "epoch_4.txt", "epoch_5.txt", "epoch_6.txt"]
arr_geometry_type_dct = {
                         "0_2_3_5": "A",
                         "0_2_4_6": "B",
                         "0_1_2_3_4_5": "C",
                         "0_1_2_3_4_5_6": "D"
                         }
all_dct = {}
all_dct_array_used_order_in_training = {}
all_dct_utt_cnt_per_epoch = {}
for file_name in file_name_lst:
    epoch_name = file_name.split(".")[0]
    print(f"====={epoch_name}=====")
    all_dct[epoch_name] = {}
    all_dct_array_used_order_in_training[epoch_name] = []
    all_dct_utt_cnt_per_epoch[epoch_name] = 0
    file_path = f"{data_dir}/{file_name}"
    utt_cnt = 0
    with open(file_path, 'r') as fr:
        lines = fr.readlines()
        for line in tqdm(lines, desc=f"{epoch_name}_processing..."):
            line = line.strip()
            line_lst = line.split()
            batch_idx = line_lst[0]
            arr_geometry_type = line_lst[1]
            utt_idx_lst = line_lst[2:]
            all_dct_array_used_order_in_training[epoch_name].append(arr_geometry_type_dct[arr_geometry_type])
            utt_cnt += len(utt_idx_lst)
            if arr_geometry_type not in all_dct[epoch_name]:
                all_dct[epoch_name][arr_geometry_type] = {}
            all_dct[epoch_name][arr_geometry_type][batch_idx] = utt_idx_lst

    print(f"all_dct_utt_cnt of {epoch_name}:{utt_cnt}")
    all_dct_utt_cnt_per_epoch[epoch_name] = utt_cnt
    print(f"all_dct_array_used_order_in_training: {all_dct_array_used_order_in_training[epoch_name][:30]}")

# stat: array geometry distribution of each epoch
stat_utts_per_epoch = {}
for epoch_name, epoch_dct in all_dct.items():
    print(f"====={epoch_name}=====")
    stat_utts_per_epoch[epoch_name] = {}
    total_cnt = 0
    for arr_geometry_type, batch_dct in epoch_dct.items():
        stat_utts_per_epoch[epoch_name][arr_geometry_type] = []
        arr_geometry_type_cnt = 0
        for batch_idx, utt_idx_lst in batch_dct.items():
            arr_geometry_type_cnt += len(utt_idx_lst)
        arr_geometry_type_percentage = arr_geometry_type_cnt / all_dct_utt_cnt_per_epoch[epoch_name]
        stat_utts_per_epoch[epoch_name][arr_geometry_type].append(arr_geometry_type_cnt)
        stat_utts_per_epoch[epoch_name][arr_geometry_type].append(all_dct_utt_cnt_per_epoch[epoch_name])
        stat_utts_per_epoch[epoch_name][arr_geometry_type].append(arr_geometry_type_percentage)
        print(f"Array geometry type: {arr_geometry_type}, "
              f"cnt: {arr_geometry_type_cnt}, "
              f"all_utt_this_epoch: {all_dct_utt_cnt_per_epoch[epoch_name]}, "
              f"percentage: {arr_geometry_type_percentage:.4f}")


# stat progress of array geometry covering utterances in training
arr_covering_dct = {}
# save_utts_per_epoch = {}
global_array_dct = {
                    "0_2_3_5": [],
                    "0_2_4_6": [],
                    "0_1_2_3_4_5": [],
                    "0_1_2_3_4_5_6": []
                }
for epoch_name, epoch_dct in all_dct.items():
    print(f"====={epoch_name}=====")
    arr_covering_dct[epoch_name] = {}
    # save_utts_per_epoch[epoch_name] = {}
    for arr_geometry_type, batch_dct in epoch_dct.items():
        arr_covering_dct[epoch_name][arr_geometry_type] = []
        # save_utts_per_epoch[epoch_name][arr_geometry_type] = []
        for batch_idx, utt_idx_lst in batch_dct.items():
            for utt_idx in utt_idx_lst:
                # save_utts_per_epoch[epoch_name][arr_geometry_type].append(utt_idx)
                global_array_dct[arr_geometry_type].append(utt_idx)
        # check set and lst
        arr_covering_dct[epoch_name][arr_geometry_type].append(len(set(global_array_dct[arr_geometry_type])))
        arr_covering_dct[epoch_name][arr_geometry_type].append(round(len(set(global_array_dct[arr_geometry_type]))/all_dct_utt_cnt_per_epoch[epoch_name], 4))
    # import pdb; pdb.set_trace()
print(f"arr_covering_dct:{arr_covering_dct}")








