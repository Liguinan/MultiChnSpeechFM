from audio_zen.dataset.base_dataset import BaseDataset
import pickle
import socket
import kaldiio
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
from torch.utils.data import Sampler
import torch.distributed as dist


def select_balanced_utterances(sorted_utt_ids, filtered_utt2dur, target_hours=10, types_per_hours=2.5):
    """
    从已排序的utterance ID列表中选择平衡的utterances

    参数:
    - sorted_utt_ids: 按duration排序的utterance ID列表
    - filtered_utt2dur: utterance ID到duration的字典映射
    - target_hours: 目标总时长（小时），默认为10小时
    - types_per_hours: 每种类型的目标时长（小时），默认为2.5小时

    返回:
    - selected_utt_ids: 最终选择的utterance ID列表
    - stats: 统计信息字典
    """

    # 转换为秒（假设duration单位为秒）
    target_seconds = target_hours * 3600
    type_target_seconds = types_per_hours * 3600

    # 按类型分类utterances
    type_lists = {
        'rev1': [],
        'rev2': [],
        'rev3': [],
        'rev4': []
    }

    # 按类型将utterances分类
    for utt_id in sorted_utt_ids:
        for type_prefix in type_lists.keys():
            if utt_id.startswith(type_prefix):
                type_lists[type_prefix].append(utt_id)
                break

    selected_utt_ids = []
    type_durations = {type_name: 0 for type_name in type_lists.keys()}
    total_duration = 0

    # 为每种类型选择utterances
    for type_name, utt_ids in type_lists.items():
        current_duration = 0
        # import pdb; pdb.set_trace()
        for utt_id in utt_ids:
            duration = filtered_utt2dur.get(utt_id, 0)
            if duration <= 0:
                continue

            # 检查是否应该结束该类型的选取
            if current_duration >= type_target_seconds:
                break

            # 添加当前utterance
            selected_utt_ids.append(utt_id)
            current_duration += duration
            type_durations[type_name] = current_duration
            total_duration += duration

            # 如果添加后超过目标时长，结束该类型的选取
            if current_duration > type_target_seconds:
                break

    # 计算统计信息
    stats = {
        'total_selected': len(selected_utt_ids),
        'total_duration_hours': total_duration / 3600,
        'type_durations': {},
        'type_counts': {}
    }

    for type_name in type_lists.keys():
        stats['type_durations'][type_name] = type_durations[type_name] / 3600
        stats['type_counts'][type_name] = len([utt for utt in selected_utt_ids if utt.startswith(type_name)])

    return selected_utt_ids, stats


class Dataset(BaseDataset):
    def __init__(
            self,
            dataset_path,
            filtered_utt_dur_range_in_seconds,
            max_nspk,
            n_mic,
            training_utt_fixed_length,
            sr,
            first_n_hours,
            crop_mode="random",
    ):
        super().__init__()
        with open(dataset_path, 'rb') as fp:
            self.info = pickle.load(fp, encoding='utf-8')
        # self.wav_list = list(self.info.keys())
        # print(f"utterances num: {len(self.wav_list)}")
        self.max_nspk = max_nspk
        self.n_mic = n_mic
        self.hostname = socket.gethostname().split(".")[0]
        print(f"Running on {self.hostname}")
        self.training_utt_fixed_length = training_utt_fixed_length
        self.sr = sr
        self.crop_mode = crop_mode

        utt2dur_dct = {}
        filtered_utt2dur = {}
        for utt_id, v in self.info.items():
            utt_dur = float(v['time_idx'][0][1])
            utt2dur_dct[utt_id] = utt_dur
            if filtered_utt_dur_range_in_seconds[0] < utt_dur < filtered_utt_dur_range_in_seconds[1]:
                filtered_utt2dur[utt_id] = utt_dur
        print(f"After filtering, the utts number is {len(filtered_utt2dur)}")

        # sort utterances according to duration for sampler sampling
        # import pdb; pdb.set_trace()
        sorted_utt_ids = sorted(filtered_utt2dur, key=lambda x: filtered_utt2dur[x])

        if  first_n_hours != -1:
            self.wav_lst, stats = select_balanced_utterances(sorted_utt_ids,
                                                         filtered_utt2dur,
                                                         target_hours=first_n_hours,
                                                         types_per_hours=first_n_hours/4)
            # import pdb; pdb.set_trace()
            print(f"first {first_n_hours} hours utterances, stats: {stats}")
        else:
            self.wav_lst = sorted_utt_ids

        self.sorted_durs = [filtered_utt2dur[utt_id] for utt_id in sorted_utt_ids]

        self.length = len(self.wav_lst)

    def __len__(self):
        return self.length

    def get_duration(self, idx):
        """获取排序后索引对应样本的duration"""
        return self.sorted_durs[idx]

    def __getitem__(self, item):
        # import pdb; pdb.set_trace()
        wav_file = self.wav_lst[item]
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

        if self.training_utt_fixed_length > 0:
            fixed_sample_length = self.training_utt_fixed_length * self.sr
            if self.crop_mode == "head":
                if wav.shape[1] >= fixed_sample_length:
                    wav = wav[:, :fixed_sample_length]
                else:
                    pad_width = fixed_sample_length - wav.shape[1]
                    wav = np.pad(wav, ((0, 0), (0, pad_width)), mode="constant")
            elif wav.shape[1] > fixed_sample_length:
                start = np.random.randint(0, wav.shape[1] - fixed_sample_length)
                wav = wav[:, start : start + fixed_sample_length]

        mc_mix_wav = wav[:self.n_mic]

        sc_reference_wav = wav[-1:, :]

        return mc_mix_wav, sc_reference_wav, directions, nspk_in_wav, mc_mix_wav.shape[1], wav_file, item


class DistributedStratifiedSampler(Sampler):
    """
    结合分布式训练和分层采样的Sampler
    支持多GPU训练和轮询分层采样
    """

    def __init__(self, dataset, batch_size=32, num_strata=2,
                 num_replicas=None, rank=None, shuffle=True, seed=0):
        """
        Args:
            dataset: 已排序的数据集
            batch_size: 每个batch的大小
            num_strata: 分层数量
            num_replicas: 分布式训练中的进程数（GPU数）
            rank: 当前进程的rank
            shuffle: 是否打乱
            seed: 随机种子
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_strata = num_strata
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        self.rng = np.random.default_rng(seed + self.epoch)

        self.dataset_size = len(dataset)

        self.usable_samples = (self.dataset_size // num_strata) * num_strata
        self.samples_per_stratum = self.usable_samples // num_strata

        self.usable_samples_per_stratum = (self.samples_per_stratum // batch_size) * batch_size
        self.usable_samples = self.usable_samples_per_stratum * num_strata

        self.batches_per_stratum = self.usable_samples_per_stratum // batch_size
        self.total_batches = self.batches_per_stratum * num_strata

        # 分配样本到strata（轮询分配）
        self._assign_samples_round_robin()
        # import pdb; pdb.set_trace()
        
        # 预计算每个batch的stratum_id映射
        self._precompute_batch_strata()
        # import pdb; pdb.set_trace()
        
        # 分布式分配：计算每个进程应该处理的batch
        self._distribute_batches()
        
        # import pdb; pdb.set_trace()

        print(f"[Rank {rank}] DistributedStratifiedSampler初始化:")
        print(f"总样本数: {self.dataset_size}")
        print(f"可用样本数: {self.usable_samples}")
        print(f"每stratum样本数: {self.usable_samples_per_stratum}")
        print(f"每stratum batch数: {self.batches_per_stratum}")
        print(f"总batch数: {self.total_batches}")

    def _assign_samples_round_robin(self):
        """轮询分配样本到strata"""
        self.strata_indices = [[] for _ in range(self.num_strata)]

        # 轮询分配前usable_samples个样本
        for i in range(self.usable_samples):
            stratum_id = i % self.num_strata
            self.strata_indices[stratum_id].append(i)
    
    def _precompute_batch_strata(self):
        """预计算每个batch对应的stratum_id"""
        self.batch_stratum_map = []
        
        # 按照轮流采样的顺序：stratum0,1,2,3,0,1,2,3,...
        for round_idx in range(self.batches_per_stratum):
            for stratum_id in range(self.num_strata):
                self.batch_stratum_map.append(stratum_id)
    
    def _distribute_batches(self):
        """分布式分配batch到各个进程"""
        # import pdb; pdb.set_trace()
        # 计算每个进程应该处理的总batch数
        self.batches_per_replica = self.total_batches // self.num_replicas
        if self.total_batches % self.num_replicas != 0:
            # 如果不能整除，给前面的进程多分配一些
            self.batches_per_replica += 1 if self.rank < self.total_batches % self.num_replicas else 0
        
        # 计算当前进程的起始和结束batch索引
        per_replica = self.total_batches // self.num_replicas
        remainder = self.total_batches % self.num_replicas
        
        if self.rank < remainder:
            self.start_batch = self.rank * (per_replica + 1)
            self.end_batch = self.start_batch + per_replica + 1
        else:
            self.start_batch = remainder * (per_replica + 1) + (self.rank - remainder) * per_replica
            self.end_batch = self.start_batch + per_replica
        
        print(f"本进程batch数: {self.batches_per_replica}")
        print(f"本进程batch范围: [{self.start_batch}, {self.end_batch})")

    def set_epoch(self, epoch):
        """
        设置epoch，用于分布式采样
        每个epoch开始时调用，确保每个epoch有不同的随机性
        """
        self.epoch = epoch
        self.rng = np.random.default_rng(self.seed + self.epoch)

    def __iter__(self):
        """生成当前进程应该处理的batch索引"""
        # import pdb; pdb.set_trace()
        
        # 为每个stratum创建当前epoch的索引顺序
        current_strata_indices = []
        for stratum_id in range(self.num_strata):
            indices = self.strata_indices[stratum_id].copy()
            
            # 如果需要打乱，在每个stratum内部打乱
            if self.shuffle:
                self.rng.shuffle(indices)
            
            current_strata_indices.append(indices)
        
        # 为每个stratum创建batch
        strata_batches = []
        for stratum_id in range(self.num_strata):
            indices = current_strata_indices[stratum_id]
            batches = []
            
            # 划分成batch
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size:
                    batches.append(batch)
            
            strata_batches.append(batches)
        
        # 按照预计算的顺序生成当前进程应该处理的batch
        for batch_idx in range(self.start_batch, self.end_batch):
            if batch_idx < self.total_batches:
                # 获取当前batch对应的stratum_id
                stratum_id = self.batch_stratum_map[batch_idx]
                
                # 获取在当前stratum中的batch索引
                stratum_batch_idx = batch_idx // self.num_strata
                
                # 确保不越界
                if stratum_batch_idx < len(strata_batches[stratum_id]):
                    yield strata_batches[stratum_id][stratum_batch_idx]

    def __len__(self):
        """返回当前进程应该处理的batch数"""
        return self.batches_per_replica

    def get_stratum_info(self, batch_idx_global):
        """获取全局batch索引对应的stratum信息"""
        # 计算batch属于哪个stratum
        if batch_idx_global < self.total_batches:
            stratum_id = self.batch_stratum_map[batch_idx_global]
            stratum_batch_idx = batch_idx_global // self.num_strata
            return stratum_id, stratum_batch_idx
        return -1, -1

def collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    # import pdb; pdb.set_trace()
    mix_, ref_, doa, spk_num, mix_length_, wav_file_, item_idx_ = zip(*data)

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

    data = {}
    data['mix'] = mixs
    data['ref'] = refs
    data['src_doa'] = torch.Tensor(np.array(doa).astype(np.float32))
    data['spk_num'] = torch.Tensor(np.array(spk_num).astype(np.float32))
    data['ori_wav_len'] = torch.Tensor(audio_frame_len).int()
    data['wav_file_lst'] = wav_file_lst
    data['item_idx'] = item_idx_

    return data