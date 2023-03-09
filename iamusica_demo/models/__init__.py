#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
"""


import torch
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
#
from .ov import OnsetsAndVelocities
from .decoder import OnsetVelocityNmsDecoder

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


# ##############################################################################
# # MODEL GETTERS
# ##############################################################################
def load_model(model, path, eval_phase=True, strict=True, device="cpu"):
    """
    """
    state_dict = torch.load(path, map_location=torch.device(device))
    model.load_state_dict(state_dict, strict=strict)
    if eval_phase:
        model.eval()
    else:
        model.train()


def get_ov_demo_model(model_path, num_mels=229, num_keys=88,
                      conv1x1_head=(200, 200), lrelu_slope=0.1, device="cpu"):
    """
    Returns a function that receives a LogMel of shape ``(mels, t)`` plus a
    threshold, and returns a decoded onset-with-vel pianoroll of shape
    ``(keys, t)``, plus the corresponding onset dataframe.
    """
    model = OnsetsAndVelocities(in_chans=2,
                                in_height=num_mels,
                                out_height=num_keys,
                                conv1x1head=conv1x1_head,
                                bn_momentum=0,
                                leaky_relu_slope=lrelu_slope,
                                dropout_drop_p=0).to(device)
    load_model(model, model_path, eval_phase=True, device=device)
    #
    decoder = OnsetVelocityNmsDecoder(
        num_keys, nms_pool_ksize=3, gauss_conv_stddev=1,
        gauss_conv_ksize=11, vel_pad_left=1, vel_pad_right=1)
    #
    def model_inf(x, pthresh=0.75):
        """
        """
        # gather onset probs, velocity estimations and decoded onsets
        with torch.no_grad():
            probs, vels = model(x.unsqueeze(0), trainable_onsets=False)
            probs = F.pad(torch.sigmoid(probs[-1]), (1, 0))
            vels = F.pad(torch.sigmoid(vels), (1, 0))
            df = decoder(probs, vels, pthresh)
        # convert decoded onsets back to probs with velocity
        probs *= 0
        probs[0][df["key"], df["t_idx"]] = torch.from_numpy(
            df["vel"].to_numpy())
        return probs[0], df

    #
    return model_inf
