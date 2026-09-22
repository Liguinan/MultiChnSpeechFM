"""
Full-utterance dataset (no random 4s crop).

Use this when you want to preserve the simulation prior that the target
speaker speaks first for the whole utterance. Training/val/test then share
the same temporal extent (modulo hop alignment in collate_fn).
"""
import numpy as np
import kaldiio

from dataset import Dataset as _CropDataset


class Dataset(_CropDataset):
    """Same as dataset.Dataset but never crops; always returns the full wav."""

    def __getitem__(self, item):
        wav_file = self.wav_lst[item]
        wav_path = self.info[wav_file]["wav_ark_path"]
        wav_info = self.info[wav_file]

        nspk_in_wav = wav_info["n_spk"]
        spk_doa = wav_info["spk_doa"]
        directions = []
        for x in range(nspk_in_wav):
            directions.append(spk_doa[x] * np.pi / 180)
        for x in range(nspk_in_wav, self.max_nspk):
            directions.append(-1)
        directions = np.array(directions).astype(np.float32)

        wav = kaldiio.load_mat(wav_path)[1].T

        mc_mix_wav = wav[: self.n_mic]
        sc_reference_wav = wav[-1:, :]

        return (
            mc_mix_wav,
            sc_reference_wav,
            directions,
            nspk_in_wav,
            mc_mix_wav.shape[1],
            wav_file,
            item,
        )
