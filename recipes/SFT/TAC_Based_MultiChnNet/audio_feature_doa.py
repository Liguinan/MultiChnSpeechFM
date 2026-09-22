import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def init_kernel(frame_len,
                frame_hop,
                num_fft=None,
                window="sqrt_hann"):
    if window != "sqrt_hann":
        raise RuntimeError("Now only support sqrt hanning window in order "
                           "to make signal perfectly reconstructed")
    if not num_fft:
        # FFT points
        fft_size = 2 ** math.ceil(math.log2(frame_len))
    else:
        fft_size = num_fft
    # window [window_length]
    window = torch.hann_window(frame_len) ** 0.5
    S_ = 0.5 * (fft_size * fft_size / frame_hop) ** 0.5
    # window_length, F, 2 (real+imag)
    # import pdb; pdb.set_trace()
    # kernel = torch.rfft(torch.eye(fft_size) / S_, 1)[:frame_len]
    # print("torch.fft.rfft")
    kernel = torch.fft.rfft(torch.eye(fft_size) / S_, dim=-1)[:frame_len]
    kernel = torch.stack((kernel.real, kernel.imag), -1)
    # 2, F, window_length
    kernel = torch.transpose(kernel, 0, 2) * window
    # 2F, 1, window_length
    kernel = torch.reshape(kernel, (fft_size + 2, 1, frame_len))
    return kernel


class STFTBase(nn.Module):
    """
    Base layer for (i)STFT
    NOTE:
        1) Recommend sqrt_hann window with 2**N frame length, because it
           could achieve perfect reconstruction after overlap-add
        2) Now haven't consider padding problems yet
    """

    def __init__(self,
                 frame_len,
                 frame_hop,
                 window="sqrt_hann",
                 num_fft=None):
        super(STFTBase, self).__init__()
        K = init_kernel(
            frame_len,
            frame_hop,
            num_fft=num_fft,
            window=window)
        self.K = nn.Parameter(K, requires_grad=False)
        self.stride = frame_hop
        self.window = window

    def freeze(self):
        self.K.requires_grad = False

    def unfreeze(self):
        self.K.requires_grad = True

    def check_nan(self):
        num_nan = torch.sum(torch.isnan(self.K))
        if num_nan:
            raise RuntimeError(
                "detect nan in STFT kernels: {:d}".format(num_nan))

    def extra_repr(self):
        return "window={0}, stride={1}, requires_grad={2}, kernel_size={3[0]}x{3[2]}".format(
            self.window, self.stride, self.K.requires_grad, self.K.shape)


class STFT(STFTBase):
    """
    Short-time Fourier Transform as a Layer
    """

    def __init__(self, *args, **kwargs):
        super(STFT, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        Accept raw waveform and output magnitude and phase
        x: input signal, N x 1 x S or N x S
        m: magnitude, N x F x T
        p: phase, N x F x T
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("Expect 2D/3D tensor, but got {:d}D".format(
                x.dim()))
        self.check_nan()
        # if N x S, reshape N x 1 x S
        if x.dim() == 2:
            x = torch.unsqueeze(x, 1)
        # N x 2F x T
        c = F.conv1d(x, self.K, stride=self.stride, padding=0)
        # N x F x T
        r, i = torch.chunk(c, 2, dim=1)
        m = (r ** 2 + i ** 2) ** 0.5
        p = torch.atan2(i, r)
        return r, i, m, p


class iSTFT(STFTBase):
    """
    Inverse Short-time Fourier Transform as a Layer
    """

    def __init__(self, *args, **kwargs):
        super(iSTFT, self).__init__(*args, **kwargs)

    def forward(self, m, p, squeeze=False):
        """
        Accept phase & magnitude and output raw waveform
        m, p: N x F x T
        s: N x C x S
        """
        if p.dim() != m.dim() or p.dim() not in [2, 3]:
            raise RuntimeError("Expect 2D/3D tensor, but got {:d}D".format(
                p.dim()))
        self.check_nan()
        # if F x T, reshape 1 x F x T
        if p.dim() == 2:
            p = torch.unsqueeze(p, 0)
            m = torch.unsqueeze(m, 0)
        r = m * torch.cos(p)
        i = m * torch.sin(p)
        # N x 2F x T
        c = torch.cat([r, i], dim=1)
        # N x 2F x T
        s = F.conv_transpose1d(c, self.K, stride=self.stride, padding=0)
        if squeeze:
            s = torch.squeeze(s)
        return s


class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: [B, F, T]
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # [B, F, T] => [B, T, F]
        x = torch.transpose(x, 1, 2)
        x = super(ChannelWiseLayerNorm, self).forward(x)
        # [B, T, F] => [B, F, T]
        x = torch.transpose(x, 1, 2)
        return x


class DFComputer(nn.Module):
    def __init__(self,
                 frame_len,
                 frame_hop,
                 in_feature,
                 NFFT,
                 speaker_feature_dim,
                 cosIPD,
                 sinIPD,
                 sr,
                 AF_premasking,
                 merge_mode='sum',
                 ):
        super(DFComputer, self).__init__()
        self.cosIPD = cosIPD
        self.sinIPD = sinIPD
        self.init_mic_pos()
        self.input_feature = in_feature
        self.spk_fea_dim = speaker_feature_dim
        self.spk_fea_merge_mode = merge_mode
        self.num_bins = NFFT
        self.epsilon = 1e-8
        self.sr = sr
        self.AF_premasking = AF_premasking

        # calculate DF dimension
        self.df_dim = 0

        self.stft = STFT(frame_len=frame_len, frame_hop=frame_hop, num_fft=frame_len)

        if 'LPS' in self.input_feature:
            self.df_dim += self.num_bins
            self.ln_LPS = ChannelWiseLayerNorm(self.num_bins)

        if 'IPD' in self.input_feature:
            # self.df_dim += self.num_bins * self.n_mic_pairs
            self.df_dim += self.num_bins
            if self.sinIPD:
                self.df_dim += self.num_bins
        if 'AF' in self.input_feature:
            self.df_dim += self.num_bins
            # self.df_dim += self.num_bins * speaker_feature_dim

    @property
    def df_dim_no_af(self):
        """LPS + IPD feature dim (without AF)."""
        dim = 0
        if 'LPS' in self.input_feature:
            dim += self.num_bins
        if 'IPD' in self.input_feature:
            dim += self.num_bins
            if self.sinIPD:
                dim += self.num_bins
        return dim

    def init_mic_pos(self):
        # The last mic postion is the averaged virtual mic position
        # in our sampled array geometry, the postion of virtual mic is the same with the 6-th microphone
        self.mic_position = np.array(
            [[-0.06375, 0], 
            [-0.031875, 0.055209], 
            [0.031875, 0.055209], 
            [0.06375, 0], 
            [0.031875, -0.055209],
            [-0.031875, -0.055209], 
             [0, 0], 
             [0, 0]])
        self.n_mic = self.mic_position.shape[0]
        distance = np.zeros([self.n_mic, self.n_mic])
        for i in range(self.n_mic):
            for j in range(self.n_mic):
                dis = math.sqrt((self.mic_position[i][0] - self.mic_position[j][0]) ** 2 
                                + (self.mic_position[i][1] - self.mic_position[j][1]) ** 2)
                distance[i][j] = dis

        self.mic_pair_distance = distance

    def update_mic_pair(self, geometry):
        mic_pairs = []
        for item in geometry:
            # idx=7 represents the averaged spectrum channel.
            mic_pairs.append([int(item), 7])
        self.mic_pairs = np.array(mic_pairs)
        self.ipd_left = [t[0] for t in mic_pairs]
        self.ipd_right = [t[1] for t in mic_pairs]
        self.n_mic_pairs = self.mic_pairs.shape[0]
    
    def compute_ipd(self, phase):
        # [B, C, F, T] -> [B, mic_pair, F, T]
        ipd = phase[:, self.ipd_left] - phase[:, self.ipd_right]
        cos_ipd = torch.cos(ipd)
        sin_ipd = torch.sin(ipd)
        return cos_ipd, sin_ipd

    def get_angle4pair(self, a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        angle = math.atan2(dy, dx)
        return angle

    def get_stv(self, angles, sndvelocity=340):
        '''
        Input: 
            angles: [B, max_nspk]
        return: 
            steering vector [Batch, n_channel, nspk, n_bins]
        '''
        dis = []
        for pair in self.mic_pairs:
            pair_d = self.mic_pair_distance[pair[0], pair[1]]
            # calcuate pair angle instead of original angle (原始角度建立在笛卡尔坐标系下，且声波先到达pair的左侧的麦克风，则cos计算结果为正)
            angle_between_face_d = angles - self.get_angle4pair(self.mic_position[pair[0]], self.mic_position[pair[1]])
            # import pdb; pdb.set_trace()
            dis_ = pair_d * torch.cos(angle_between_face_d)
            dis.append(dis_)
        # mic_pair * [B, max_spk_num] -> [B, mic_pair, max_spk_num]
        distances = torch.stack(dis, dim=1)  # [64, 9, 3]
        deltas = distances / sndvelocity * self.sr
        stv_real = []
        stv_imag = []
        for f in range(self.num_bins):
            stv_real.append(torch.cos(deltas * math.pi * f / (self.num_bins - 1)))
            stv_imag.append(torch.sin(deltas * math.pi * f / (self.num_bins - 1)))
        # self.num_bins * [B, mic_pair, max_spk_num] -> [self.num_bins, B, mic_pair, max_spk_num]
        stv_real = torch.stack(stv_real)
        stv_imag = torch.stack(stv_imag)
        # 保持功率模值平方为1 
        stv_real /= math.sqrt(self.n_mic_pairs)
        stv_imag /= math.sqrt(self.n_mic_pairs)
        # [self.num_bins, B, mic_pair, max_spk_num] -> [B, mic_pair, max_spk_num, self.num_bins]
        stv_real = stv_real.permute((1, 2, 3, 0))
        stv_imag = stv_imag.permute((1, 2, 3, 0))
        return stv_real, stv_imag

    def get_AF(self, stv_real, stv_imag, ipd4AF_real, ipd4AF_imag, spk_num):
        B, m, F, T = ipd4AF_imag.shape
        rlt_rr_ein = torch.einsum('bmcf,bmfk->bcfk', (stv_real, ipd4AF_real))
        rlt_ii_ein = torch.einsum('bmcf,bmfk->bcfk', (stv_imag, ipd4AF_imag))
        # -> [B, max_spk_num, F, T]
        AF = rlt_rr_ein + rlt_ii_ein 
        AFs = []
        # import pdb; pdb.set_trace()
        for b in range(B):
            nspk = spk_num[b]
            # only extract target speaker AF (idx=0) 
            _AF_tgt = AF[b, 0] 
            if self.AF_premasking:
                for idx in range(1, nspk.int()):
                    _AF_tgt[_AF_tgt < AF[b, idx]] = 0
            _AF_tgt = (_AF_tgt - torch.mean(_AF_tgt, dim=-1, keepdim=True)) \
                      / (torch.std(_AF_tgt, dim=-1, keepdim=True) + 1e-8)
            if self.spk_fea_dim == 2:
                if nspk == 1:
                    _AF_intf = torch.zeros_like(_AF_tgt)
                else:
                    _AF_intfs = []
                    for idx in range(1, nspk):
                        _AF_intf = AF[b, idx]
                        for j in range(1, nspk):
                            if idx != j:
                                _AF_intf[_AF_intf < AF[b, j]] = 0
                        _AF_intfs.append(_AF_intf)
                    if self.spk_fea_merge_mode == 'sum':
                        _AF_intf = torch.sum(torch.stack(_AF_intfs, dim=-1), dim=-1)
                    elif self.spk_fea_merge_mode == 'ave':
                        _AF_intf = torch.mean(torch.stack(_AF_intfs, dim=-1), dim=-1)
                    elif self.spk_fea_merge_mode == 'closest':
                        raise NotImplementedError 
                    _AF_intf = (_AF_intf - torch.mean(_AF_intf, dim=-1, keepdim=True)) \
                               / (torch.std(_AF_intf, dim=-1, keepdim=True) + 1e-8)

                AFs.append(torch.cat((_AF_tgt, _AF_intf), dim=0))
            else:
                AFs.append(_AF_tgt)
        # B * [F, T] -> [B, F, T]
        AF = torch.stack(AFs, dim=0)
        AF = AF.view(B, self.num_bins * self.spk_fea_dim, T)
        return AF

    def _stft_and_base(self, x, geometry, ln_lps=None):
        """
        Shared STFT + LPS + IPD.
        ln_lps: optional external LPS LayerNorm; default self.ln_LPS.
        """
        self.update_mic_pair(geometry)

        B, C, t = x.shape
        x = x.reshape(-1, t)
        real, imag, mag, phase = self.stft(x)
        _, F, T = phase.shape
        phase = phase.view(B, C, F, T)
        mag = mag.view(B, C, F, T)

        real_avg = real.view(B, C, F, T)[:, geometry, :, :].mean(dim=1)
        imag_avg = imag.view(B, C, F, T)[:, geometry, :, :].mean(dim=1)
        mag_avg = (real_avg ** 2 + imag_avg ** 2) ** 0.5
        phase_avg = torch.atan2(imag_avg, real_avg)

        mag = torch.cat([mag, mag_avg.unsqueeze(dim=1)], dim=1)
        phase = torch.cat([phase, phase_avg.unsqueeze(dim=1)], dim=1)

        cos_ipd, sin_ipd = None, None
        if 'IPD' in self.input_feature:
            cos_ipd, sin_ipd = self.compute_ipd(phase)

        raw_lps = None
        if 'LPS' in self.input_feature:
            raw_lps = torch.log(mag_avg ** 2 + self.epsilon)

        def _build_base(ln_mod):
            df = []
            if 'LPS' in self.input_feature:
                lps = ln_mod(raw_lps)
                lps = lps.unsqueeze(dim=1).expand(-1, len(geometry), -1, -1)
                df.append(lps)
            if 'IPD' in self.input_feature:
                df.append(cos_ipd)
                if self.sinIPD:
                    df.append(sin_ipd)
            return torch.cat(df, dim=2)

        ln_mod = ln_lps if ln_lps is not None else self.ln_LPS
        base_df = _build_base(ln_mod)

        ipd4AF = phase[:, self.ipd_left] - phase[:, self.ipd_right]
        ipd4AF_real = torch.cos(ipd4AF) / self.n_mic_pairs
        ipd4AF_imag = torch.sin(ipd4AF) / self.n_mic_pairs
        return base_df, mag, phase, ipd4AF_real, ipd4AF_imag

    def forward_base(self, x, geometry, ln_lps=None):
        """LPS + IPD only, no AF. Optional external ln_lps."""
        return self._stft_and_base(x, geometry, ln_lps=ln_lps)

    def forward_base_dual_ln(self, x, geometry, ln_doa, ln_sep):
        """
        One STFT; build two base_df with different LPS LayerNorms
        (DOA path vs separation path).
        """
        self.update_mic_pair(geometry)

        B, C, t = x.shape
        x_flat = x.reshape(-1, t)
        real, imag, mag, phase = self.stft(x_flat)
        _, F, T = phase.shape
        phase = phase.view(B, C, F, T)
        mag = mag.view(B, C, F, T)

        real_avg = real.view(B, C, F, T)[:, geometry, :, :].mean(dim=1)
        imag_avg = imag.view(B, C, F, T)[:, geometry, :, :].mean(dim=1)
        mag_avg = (real_avg ** 2 + imag_avg ** 2) ** 0.5
        phase_avg = torch.atan2(imag_avg, real_avg)

        mag = torch.cat([mag, mag_avg.unsqueeze(dim=1)], dim=1)
        phase = torch.cat([phase, phase_avg.unsqueeze(dim=1)], dim=1)

        cos_ipd, sin_ipd = None, None
        if 'IPD' in self.input_feature:
            cos_ipd, sin_ipd = self.compute_ipd(phase)
        raw_lps = torch.log(mag_avg ** 2 + self.epsilon)

        def _build_base(ln_mod):
            df = []
            if 'LPS' in self.input_feature:
                lps = ln_mod(raw_lps)
                lps = lps.unsqueeze(dim=1).expand(-1, len(geometry), -1, -1)
                df.append(lps)
            if 'IPD' in self.input_feature:
                df.append(cos_ipd)
                if self.sinIPD:
                    df.append(sin_ipd)
            return torch.cat(df, dim=2)

        base_doa = _build_base(ln_doa)
        base_sep = _build_base(ln_sep)
        ipd4AF = phase[:, self.ipd_left] - phase[:, self.ipd_right]
        ipd4AF_real = torch.cos(ipd4AF) / self.n_mic_pairs
        ipd4AF_imag = torch.sin(ipd4AF) / self.n_mic_pairs
        return base_doa, base_sep, mag, phase, ipd4AF_real, ipd4AF_imag

    def compute_AF(self, directions, ipd4AF_real, ipd4AF_imag, nspk, geometry):
        """
        Compute AF from estimated / GT directions.
        directions: [B, max_nspk] in radians
        Returns: [B, len(geometry), num_bins * spk_fea_dim, T]
        """
        if 'AF' not in self.input_feature:
            raise RuntimeError("AF not enabled in input_feature")
        stv_real, stv_imag = self.get_stv(directions)
        AF = self.get_AF(stv_real, stv_imag, ipd4AF_real, ipd4AF_imag, nspk)
        AF = AF.unsqueeze(dim=1).expand(-1, len(geometry), -1, -1)
        return AF

    def forward_with_directions(self, x, directions, nspk, geometry):
        """Full feature [LPS, IPD, AF] using given directions."""
        base_df, mag, phase, ipd4AF_real, ipd4AF_imag = self._stft_and_base(x, geometry)
        AF = self.compute_AF(directions, ipd4AF_real, ipd4AF_imag, nspk, geometry)
        audio_fea = torch.cat([base_df, AF], dim=2)
        return audio_fea, mag, phase

    def forward(self, x, directions, nspk, geometry):
        """
        Compute audio features (compatible with original DFComputer.forward).
        [0] x - multi-channel mixture wav, shape: [B, C+1, t]
        [1] directions - all speakers' directions relative to microphone array, shape: [B, spk_num]
        [2] spk_num - actual speaker number in current wav shape: [B]
        [3] geometry - array geometry of each batch, shape: [selected_microphone_num_in_array_geometry]
        :return: spatial features & directional features
        """
        base_df, mag, phase, ipd4AF_real, ipd4AF_imag = self._stft_and_base(x, geometry)
        if 'AF' in self.input_feature:
            AF = self.compute_AF(directions, ipd4AF_real, ipd4AF_imag, nspk, geometry)
            df = torch.cat([base_df, AF], dim=2)
        else:
            df = base_df
        return df, mag, phase

if __name__ == '__main__':
    pass