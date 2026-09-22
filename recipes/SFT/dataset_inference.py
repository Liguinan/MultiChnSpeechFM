import random

import numpy as np
from joblib import Parallel, delayed
from scipy import signal
from tqdm import tqdm

from audio_zen.acoustics.feature import (
    is_clipped,
    load_wav,
    norm_amplitude,
    subsample,
    tailor_dB_FS,
)
from audio_zen.dataset.base_dataset import BaseDataset
from audio_zen.utils import expand_path
import pickle
import socket
import kaldiio
from torch.nn.utils.rnn import pad_sequence
import torch


class Dataset(BaseDataset):
    def __init__(
            self,
            dataset_path,
            max_nspk,
            n_mic
    ):
        super().__init__()
        with open(dataset_path, 'rb') as fp:
            self.info = pickle.load(fp, encoding='utf-8')
        self.wav_list = list(self.info.keys())
        print(f"utterances num: {len(self.wav_list)}")
        self.max_nspk = max_nspk
        self.n_mic = n_mic
        self.hostname = socket.gethostname().split(".")[0]
        print(f"Running on {self.hostname}")

        self.length = len(self.wav_list)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        wav_file = self.wav_list[item]
        wav_path = self.info[wav_file]["wav_ark_path"]
        wav_info = self.info[wav_file]

        nspk_in_wav = wav_info['n_spk']
        spk_doa = wav_info['spk_doa']
        directions = []
        for x in range(nspk_in_wav):
            directions.append(spk_doa[x] * np.pi / 180)
        for x in range(nspk_in_wav, self.max_nspk):
            directions.append(-1)
        directions = np.array(directions).astype(np.float32)
        
        # [7chn mixture wav, clean wav, reverb_clean wav]
        # reverb_clean wav align with 1chn of 7chn mixture wav
        wav = kaldiio.load_mat(wav_path)[1].T 

        mc_mix_wav = wav[:self.n_mic]
        sc_reference_wav = wav[-1:, :]

        return mc_mix_wav, sc_reference_wav, directions, nspk_in_wav, mc_mix_wav.shape[1], wav_file


def collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    mix_, ref_, doa, spk_num, mix_length_, wav_file_ = zip(*data)

    mixs = pad_sequence([torch.from_numpy(mix.copy()).transpose(0, 1) for mix in mix_],  padding_value=0, batch_first=True).transpose(1, 2)
    refs = pad_sequence([torch.from_numpy(ref.copy()).transpose(0, 1) for ref in ref_],  padding_value=0, batch_first=True).transpose(1, 2)
    # For stft frames calculation
    max_len = max(mix_length_)
    max_len = max_len // 256 * 256  # 256 is HOP SIZE 
    mixs = mixs[:, :, :max_len]
    refs = refs[:, :, :max_len]

    # frames = 1 + floor((L - 512) / 256) = L//256 - 2 + 1 = L//256 - 1
    audio_frame_len = [int(mix_length // 256) - 1 for mix_length in mix_length_]

    wav_file_lst = [wav_file for wav_file in wav_file_]

    # import pdb; pdb.set_trace()
    data = {}
    data['mix'] = mixs
    data['ref'] = refs
    data['src_doa'] = torch.Tensor(np.array(doa).astype(np.float32))
    data['spk_num'] = torch.Tensor(np.array(spk_num).astype(np.float32))
    data['ori_wav_len'] = torch.Tensor(audio_frame_len).int()
    data['wav_file_lst'] = wav_file_lst

    return data