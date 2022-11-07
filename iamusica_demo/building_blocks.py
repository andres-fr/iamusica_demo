#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
"""


import torch
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB


# ##############################################################################
# # NN INITIALIZATION
# ##############################################################################
def init_weights(module, init_fn=torch.nn.init.kaiming_normal,
                 bias_val=0.0, verbose=False):
    """
    :param init_fn: initialization function, such that ``init_fn(weight)``
      modifies in-place the weight values. If ``None``, found weights won't be
      altered
    :param float bias_val: Any module with biases will initialize them to this
      constant value

    Usage example, inside of any ``torch.nn.Module.__init__`` method:

    if init_fn is not None:
            self.apply(lambda module: init_weights(module, init_fn, 0.0))

    Apply is applied recursively to any submodule inside, so this works.
    """
    if isinstance(module, (torch.nn.Linear,
                           torch.nn.Conv1d,
                           torch.nn.Conv2d)):
        if init_fn is not None:
            init_fn(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(bias_val)
    else:
        if verbose:
            print("init_weights: ignored module:", module.__class__.__name__)


# ##############################################################################
# # BASIC BLOCKS
# ##############################################################################
def get_relu(leaky_slope=None):
    """
    """
    if leaky_slope is None:
        result = torch.nn.ReLU(inplace=True)
    else:
        result = torch.nn.LeakyReLU(leaky_slope, inplace=True)
    return result


class Permuter(torch.nn.Module):
    """
    """

    def __init__(self, *permutation):
        """
        """
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        """
        """
        return x.permute(self.permutation)


class SubSpectralNorm(torch.nn.Module):
    """
    https://arxiv.org/pdf/2103.13620.pdf
    """

    def __init__(self, C, F, S, momentum=0.1, eps=1e-5):
        """
        :param C: Channels in batch ``(N, C, F, T)``
        :param S: Number of subbands such that ``(N, C*S, F//S, T)`` is regarded
        """
        super().__init__()
        self.S = S
        self.eps = eps
        self.bn = torch.nn.BatchNorm2d(C * S, momentum=momentum)
        assert divmod(F, S)[1] == 0, "S must divide F exactly!"

    def forward(self, x):
        """
        """
        # x: input features with shape {N, C, F, T}
        # S: number of sub-bands
        N, C, F, T = x.size()
        x = x.reshape(N, C * self.S, F // self.S, T)
        x = self.bn(x)
        return x.reshape(N, C, F, T)


class Nms1d(torch.nn.Module):
    """
    PT-compatible NMS, 1-dimensional along the last axis. Note that any
    non-zero entry that equals the maximum among the ``pool_ksize`` vicinity
    is considered a maximum. This includes if multiple maxima are present in
    a vicinity (even if disconnected), and particularly if all values are equal
    and non-zero
    """

    def __init__(self, pool_ksize=3):
        """
        """
        super().__init__()
        self.nms_pool = torch.nn.MaxPool1d(
            pool_ksize, stride=1, padding=pool_ksize // 2, ceil_mode=False)

    def forward(self, onset_preds, thresh=None):
        """
        :param onset_preds: Batch of shape ``(b, chans, t)``
        :param thresh: Any values below this will also be zeroed out
        """
        x = self.nms_pool(onset_preds)
        x = (onset_preds == x)
        x = onset_preds * x
        if thresh is not None:
            x = x * (x >= thresh)
        return x


# #############################################################################
# # CONTEXT-AWARE MODULE
# #############################################################################
class SELayer(torch.nn.Module):
    """
    Squeeze-excitation module
    """
    def __init__(self, in_chans, hidden_chans=None, out_chans=None,
                 bn_momentum=0.1):
        super().__init__()
        if hidden_chans is None:
            hidden_chans = in_chans // 4
        if out_chans is None:
            out_chans = in_chans
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)  # output a scalar per ch
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_chans, hidden_chans, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_chans, out_chans, bias=True),
            torch.nn.Sigmoid()
        )

    def set_biases(self, val=0):
        """
        """
        self.apply(lambda module: init_weights(
            module, init_fn=None, bias_val=val))

    def forward(self, x):
        """
        :param x: Input batch of shape ``(b, ch, h, w)``
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x)[:, :, 0, 0]  # (b, c)
        y = self.fc(y)[:, :, None, None]
        return y  # (b, c, 1, 1)


class ContextAwareModule(torch.nn.Module):
    """
    Context-Aware Module adapted for spectrograms from
    https://arxiv.org/pdf/1910.12223.pdf
    """
    def __init__(self,
                 in_chans, hdc_chans=None, se_bottleneck=None,
                 ksizes=((3, 5), (3, 5), (3, 5), (3, 5)),
                 dilations=((1, 1), (1, 2), (1, 3), (1, 4)),
                 paddings=((1, 2), (1, 4), (1, 6), (1, 8)),
                 bn_momentum=0.1):
        """
        """
        super().__init__()
        #
        assert len(ksizes) == len(dilations) == len(paddings), \
            "ksizes, dilations and paddings must have same length!"
        num_convs = len(ksizes)
        #
        if hdc_chans is None:
            hdc_chans = in_chans // num_convs
        hdc_out_chans = hdc_chans * num_convs
        if se_bottleneck is None:
            se_bottleneck = in_chans // 4
        #
        self.se = SELayer(in_chans, se_bottleneck, hdc_out_chans, bn_momentum)
        #
        self.hdcs = torch.nn.ModuleList(
            [torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_chans, hdc_chans, stride=1,
                    kernel_size=ks, dilation=dil, padding=pad,
                    bias=False),
                torch.nn.BatchNorm2d(hdc_chans, momentum=bn_momentum),
                torch.nn.ReLU(inplace=True))
             for ks, dil, pad in zip(ksizes, dilations, paddings)])
        #
        self.skip = torch.nn.Sequential(
            torch.nn.Conv2d(in_chans, hdc_out_chans, kernel_size=1,
                            bias=False),
            torch.nn.BatchNorm2d(hdc_out_chans, momentum=bn_momentum),
            torch.nn.ReLU(inplace=True))

    def forward(self, x):
        """
        """
        se_att = self.se(x)  # (b, hdc, 1, 1)
        skip = self.skip(x)  # (b, hdc, h, w)
        hdc = torch.cat([hdc(x) for hdc in self.hdcs], dim=1)  # (b, hdc, h, w)
        #
        x = skip + (hdc * se_att)
        return x


# ##############################################################################
# # RESHAPER CONVS
# ##############################################################################
def conv1x1net(hid_chans, bn_momentum=0.1, last_layer_bn_relu=False,
               dropout_drop_p=None, leaky_relu_slope=None,
               kernel_width=1):
    """
    """
    assert (kernel_width % 2) == 1, "Only odd kwidth supported!"
    wpad  =kernel_width // 2
    #
    result = torch.nn.Sequential()
    n_layers = len(hid_chans) - 1
    for i, (h_in, h_out) in enumerate(zip(hid_chans[:-1],
                                          hid_chans[1:]), 1):
        if (i < n_layers) or ((i == n_layers) and last_layer_bn_relu):
            result.append(torch.nn.Conv2d(h_in, h_out, (1, kernel_width),
                                          padding=(0, wpad), bias=False))
            result.append(torch.nn.BatchNorm2d(h_out, momentum=bn_momentum))
            result.append(get_relu(leaky_relu_slope))
            if dropout_drop_p is not None:
                result.append(torch.nn.Dropout(dropout_drop_p, inplace=False))
        else:
            result.append(torch.nn.Conv2d(h_in, h_out, (1, kernel_width),
                                          padding=(0, wpad), bias=True))
    #
    return result


class DepthwiseConv2d(torch.nn.Module):
    """
    For input spectrogram ``(b, ch_in, h_in, t)``, the DepthwiseConv can be
    implemented as follows (assuming ``same`` padding):
    ``torch.nn.Conv2d(ch_in, ch_out*K, (h_in, kt), groups=ch_in)`` followed
    by ``squeeze(2)`` and ``reshape(b, K, ch_out, t)``. This is so because the
    convolution yields``(b, K*ch_out, 1, t)``.

    Note that the conv filter has dimensionality ``(K*ch_out, 1, h_in, 1)``,
    but each of the ``K`` segments is applied to a separate channel, due to
    having ``groups=ch_in``.

    More info: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """

    def __init__(self, ch_in, ch_out, h_in, h_out, kernel_width=1, bias=True):
        """
        """
        super().__init__()
        self.ch_in, self.ch_out = ch_in, ch_out
        self.h_in, self.h_out = h_in, h_out
        #
        assert (kernel_width % 2) == 1, "Only odd kwidth supported!"
        self.conv = torch.nn.Conv2d(
            ch_in, ch_out * h_out,
            (h_in, kernel_width), padding=(0, kernel_width // 2), groups=ch_in,
            bias=bias)

    def forward(self, x):
        """
        :param x: Batch tensor of shape ``(b, ch_in, h_in, t)``.
        :returns: Batch tensor of shape ``(b, ch_out, h_out, t)``
        """
        b, ch_in, h_in, t = x.shape
        x = self.conv(x)
        x = x.squeeze(2).reshape(b, self.ch_out, self.h_out, -1)
        return x


# ##############################################################################
# # AD-HOC CONVOLVERS
# ##############################################################################
class GaussianBlur1d(torch.nn.Module):
    """
    Performs 1D gaussian convolution along last dimension of ``(b, c, t)``
    tensors.
    """

    @staticmethod
    def gaussian_1d_kernel(ksize=15, stddev=3.0, mean_offset=0, rbf=False,
                           dtype=torch.float32):
        """
        :param mean_offset: If 0, the mean of the gaussian is at the center of
          the kernel. So peaks at index ``t`` will appear at idx ``t+offset``
          when convolved
        :param rbf: If true, max(kernel)=1 instead of sum(kernel)=1
        """
        x = torch.linspace(-(ksize - 1) / 2., (ksize - 1) / 2., ksize, dtype=dtype)
        x += mean_offset
        x.div_(stddev).pow_(2).mul_(-0.5).exp_()
        if not rbf:
            x.div_(x.sum())
        return x

    def __init__(self, num_chans, ksize=15, stddev=3.0, mean_offset=0,
                 rbf=False):
        """
        """
        assert ksize % 2 == 1, "Only odd ksize supported!"
        super().__init__()
        self.ksize = ksize
        self.stddev = stddev
        self.blur_fn = torch.nn.Conv1d(
            num_chans, num_chans, ksize, padding=ksize // 2,
            groups=num_chans, bias=False)
        #
        with torch.no_grad():
            # create 1d gaussian kernel and reshape to match the conv1d weight
            kernel = self.gaussian_1d_kernel(
                ksize, stddev, rbf=rbf, mean_offset=mean_offset,
                dtype=self.blur_fn.weight.dtype)
            kernel = kernel[None, :].repeat(num_chans, 1).unsqueeze(1)
            # assign kernel to weight
            self.blur_fn.weight[:] = kernel

    def forward(self, x):
        """
        """
        x = self.blur_fn(x)
        return x


# ##############################################################################
# # LOGMEL
# ##############################################################################
class TorchWavToLogmelDemo(torch.nn.Module):
    """
    Torch version of WavToLogmel, plus convenience DB offset and freq shift
    functionality for real-time demo. Much faster, results differ slightly.
    Since this is a torch Module, can be sent ``.to("cuda")`` in order
    to admit CUDA tensors.
    """
    def __init__(self, samplerate, winsize, hopsize, n_mels,
                 mel_fmin=50, mel_fmax=8_000, window_fn=torch.hann_window):
        """
        :param samplerate: Expected audio input samplerate.
        :param winsize: Window size for the STFT (and mel).
        :param hopsize: Hop size for the STFT (and mel).
        :param stft_window: Windowing function for the STFT.
        :param n_mels: Number of mel bins.
        :param mel_fmin: Lowest mel bin, in Hz.
        :param mel_fmax: Highest mel bin, in Hz.
        """
        super().__init__()
        self.melspec = MelSpectrogram(
            samplerate, winsize, hop_length=hopsize,
            f_min=mel_fmin, f_max=mel_fmax, n_mels=n_mels,
            power=2, window_fn=window_fn)
        self.to_db = AmplitudeToDB(stype="power", top_db=80.0)
        # run melspec once, otherwise produces NaNs!
        self.melspec(torch.rand(winsize * 10))
        #
        self.samplerate = samplerate
        self.winsize = winsize
        self.hopsize = hopsize
        self.n_mels = n_mels

    def __call__(self, wav_arr, db_offset=0, shift_bins=0):
        """
        :param wav_arr: Float tensor array of either 1D or ``(chans, time)``
        :param db_offset: This constant will be added to the logmel. Statistics
          for a sample maestro logmel are: range ``(-40, 40)``, median ``-5``.
        :param shift_bins: The ``ith`` row becomes ``i+shift``.
        :returns: log-mel spectrogram of shape ``(n_mels, t)``
        """
        with torch.no_grad():
            mel = self.melspec(wav_arr)
            log_mel = self.to_db(mel)
            if db_offset != 0:
                log_mel += db_offset
            if shift_bins != 0:
                assert abs(shift_bins) < self.n_mels, \
                    "Shift bins must be less than num mels!"
                result = torch.full_like(log_mel, fill_value=log_mel.min())
                if shift_bins > 0:
                    result[shift_bins:, :] = log_mel[:-shift_bins, :]
                elif shift_bins < 0:
                    result[:shift_bins, :] = log_mel[-shift_bins:, :]
                log_mel = result
        return log_mel
