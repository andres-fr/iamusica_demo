#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
"""


import pandas as pd
import torch
import torch.nn.functional as F
#
from .building_blocks import Nms1d, GaussianBlur1d


# ##############################################################################
# # ONSET+VELOCITY DECODERS
# ##############################################################################
class OnsetVelocityNmsDecoder(torch.nn.Module):
    """
    Modification of ``OnsetNmsdecoder``, that also processes velocities. Given
    a pianoroll with detected onset probabilites, and an analogous roll with
    predicted velocities:
    1. Detects onsets in the same way as ``OnsetNmsdecoder``
    2. Reads the velocity at the detected onsets from the given velocity maps
    3. Returns onset positions, probabilities and velocities
    """

    def __init__(self, num_keys, nms_pool_ksize=3, gauss_conv_stddev=None,
                 gauss_conv_ksize=None, vel_pad_left=1, vel_pad_right=1):
        """
        :param num_keys: Expected input to forward is ``(b, num_keys, t)``.
        :param gauss_conv_stddev: If given
        :param gauss_conv_ksize: Unused if stddev is not given. If given, a
          default ksize of ``7*stddev`` will be taken, but here we can provide
          a custom ksize (sometimes needed since odd ksize is required).
        :param vel_pad_left: When checking the predicted velocity, how many
         indexes to the left to the peak are regarded (average of all regarded
         entries is computed).
        :param vel_pad_right: See ``vel_pad_left``.
        """
        super().__init__()
        self.num_keys = num_keys
        self.nms1d = Nms1d(nms_pool_ksize)
        #
        self.blur = gauss_conv_stddev is not None
        if self.blur:
            if gauss_conv_ksize is None:
                gauss_conv_ksize = round(gauss_conv_stddev * 7)
            self.gauss1d = GaussianBlur1d(
                num_keys, gauss_conv_ksize, gauss_conv_stddev)
        #
        self.vel_pad_left = vel_pad_left
        self.vel_pad_right = vel_pad_right

    @staticmethod
    def read_velocities(velmap, batch_idxs, key_idxs, t_idxs,
                        pad_l=0, pad_r=0):
        """
        Given:
        1. A tensor of shape ``(b, k, t)``
        2. Indexes corresponding to points in the tensor
        3. Potential span to the left and right of points across the t dim.
        This method reads and returns the corresponding points in the tensor.
        If spans are given, the results are averaged for each span.
        """
        assert pad_l >= 0, "Negative padding not allowed!"
        assert pad_r >= 0, "Negative padding not allowed!"
        # if we read extra l/r, pad to avoid OOB (reflect to retain averages)
        if (pad_l > 0) or (pad_r > 0):
            velmap = F.pad(velmap, (pad_l, pad_r), mode="reflect")
        #
        total_readings = pad_l + pad_r + 1
        result = velmap[batch_idxs, key_idxs, t_idxs]
        for delta in range(1, total_readings):
            result += velmap[batch_idxs, key_idxs, t_idxs + delta]
        result /= total_readings
        return result

    def forward(self, onset_probs, velmap, pthresh=None):
        """
        :param onset_probs: Tensor of shape ``(b, keys, t)`` expected to
          contain onset probabilities
        :param velmap: Velocity map of same shape as onset_probs, containing
          the predicted velocity for each given entry.
        :param pthresh: Any probs below this value won't be regarded.

        """
        assert 0 <= onset_probs.min() <= onset_probs.max() <= 1, \
            "Onset probs expected to contain probabilities in range [0, 1]!"
        assert onset_probs.shape == velmap.shape, \
            "Onset probs and velmap must have same shape!"
        # perform NMS on onset probs
        with torch.no_grad():
            # optional blur
            if self.blur:
                prev_max = onset_probs.max()
                if prev_max > 0:
                    onset_probs = self.gauss1d(onset_probs)
            onset_probs = self.nms1d(onset_probs, pthresh)
        # extract NMS indexes and prob values
        bbb, kkk, ttt = onset_probs.nonzero(as_tuple=True)
        ppp = onset_probs[bbb, kkk, ttt]
        # extract velocity readings. Reflect pad to avoid OOB and retain avgs
        vvv = self.read_velocities(velmap, bbb, kkk, ttt,
                                   self.vel_pad_left, self.vel_pad_right)
        # create dataframe and return
        df = pd.DataFrame(
            {"batch_idx": bbb.cpu(), "key": kkk.cpu(), "t_idx": ttt.cpu(),
             "prob": ppp.cpu(), "vel": vvv.cpu()})
        return df
