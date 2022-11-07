#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
"""


import torch
import torch.nn.functional as F
import pandas as pd
#
from .building_blocks import init_weights, get_relu
from .building_blocks import SubSpectralNorm, Permuter, Nms1d, GaussianBlur1d
from .building_blocks import ContextAwareModule, DepthwiseConv2d, conv1x1net


# ##############################################################################
# # ONSETS+VELOCITY
# ##############################################################################
class OnsetVelocity_2(torch.nn.Module):
    """
    """

    @staticmethod
    def get_cam_stage(in_chans, num_bins, conv1x1=(400, 200, 100),
                      num_cam_bottlenecks=3, cam_hdc_chans=3, cam_se_bottleneck=8,
                      cam_ksizes=((3, 21), (3, 17), (3, 13), (3, 9)),
                      cam_dilations=((1, 1), (1, 2), (1, 3), (1, 4)),
                      cam_paddings=((1, 10), (1, 16), (1, 18), (1, 16)),
                      bn_momentum=0.1, leaky_relu_slope=0.1, dropout_p=0.1,
                      summary_width=3, conv1x1_kw=1):
        """
        """
        cam_out_chans = cam_hdc_chans * len(cam_ksizes)
        cam = torch.nn.Sequential(
            torch.nn.Conv2d(in_chans, cam_out_chans, (1, 1),
                            padding=(0, 0), bias=False),
            torch.nn.BatchNorm2d(cam_out_chans, momentum=bn_momentum),
            get_relu(leaky_relu_slope),
            *[torch.nn.Sequential(
                ContextAwareModule(
                    cam_out_chans, cam_hdc_chans, cam_se_bottleneck,
                    cam_ksizes, cam_dilations, cam_paddings, bn_momentum),
                torch.nn.BatchNorm2d(cam_out_chans, momentum=bn_momentum),
                get_relu(leaky_relu_slope))
              for _ in range(num_cam_bottlenecks)],
            # output of this module is shape (b, conv1x1[0], in_bins, t)
            torch.nn.Conv2d(
                cam_out_chans, conv1x1[0], (num_bins, summary_width),
                padding=(0, 1), bias=False),
            torch.nn.BatchNorm2d(conv1x1[0], momentum=bn_momentum),
            get_relu(leaky_relu_slope),
            # output is (b, 1, out_chans, t)
            conv1x1net((*conv1x1, num_bins), bn_momentum,
                       last_layer_bn_relu=False,
                       dropout_drop_p=dropout_p,
                       leaky_relu_slope=leaky_relu_slope,
                       kernel_width=conv1x1_kw),
            Permuter(0, 2, 1, 3))
        #
        return cam

    def __init__(self, in_bins, out_bins, bn_momentum=0.1,
                 conv1x1=(400, 200, 100),
                 init_fn=torch.nn.init.kaiming_normal_,
                 se_init_bias=1.0, dropout_drop_p=0.1,
                 leaky_relu_slope=0.1):
        """
        """
        super().__init__()

        #
        in_chans = 2
        # #
        stem_num_cam_bottlenecks = 3
        stem_hdc_chans = 4
        stem_se_bottleneck = 8
        #
        stem_ksizes=((3, 5), (3, 5), (3, 5), (3, 5), (3, 5))  ### added dil=5
        stem_dilations=((1, 1), (1, 2), (1, 3), (1, 4), (1, 5))
        stem_paddings=((1, 2), (1, 4), (1, 6), (1, 8), (1, 10))
        stem_inner_chans = stem_hdc_chans * len(stem_ksizes)
        stem_out_chans = stem_inner_chans  ### * 2
        # #
        cam_num_bottlenecks = 3
        cam_hdc_chans = 4
        cam_se_bottleneck = 8
        #
        cam_ksizes=((1, 21), (1, 17), (1, 13), (1, 9))
        cam_dilations=((1, 1), (1, 2), (1, 3), (1, 4))
        cam_paddings=((0, 10), (0, 16), (0, 18), (0, 16))
        cam_out_chans = cam_hdc_chans * len(cam_ksizes)
        # #
        num_refiner_stages = 3
        refiner_num_bottlenecks = 2
        refiner_hdc_chans = 3
        refiner_se_bottleneck = 8
        #
        refiner_ksizes=((1, 9), (1, 7), (1, 5))
        refiner_dilations=((1, 1), (1, 2), (1, 3))
        refiner_paddings=((0, 4), (0, 6), (0, 6))
        refiner_out_chans = refiner_hdc_chans * len(refiner_ksizes)
        #
        self.specnorm = SubSpectralNorm(2, in_bins, in_bins, bn_momentum)
        self.stem = torch.nn.Sequential(
            # lift in chans into stem chans
            torch.nn.Conv2d(
                in_chans, stem_inner_chans, (3, 3), padding=(1, 1), bias=False),
            SubSpectralNorm(stem_inner_chans, in_bins, in_bins, bn_momentum),
            get_relu(leaky_relu_slope),
            # series of stem CAM modules. Output: (b, stem_inner_chans, mels, t)
            *[torch.nn.Sequential(
                ContextAwareModule(
                    stem_inner_chans, stem_hdc_chans, stem_se_bottleneck,
                    stem_ksizes, stem_dilations, stem_paddings, bn_momentum),
                SubSpectralNorm(stem_inner_chans, in_bins, in_bins, bn_momentum),
                get_relu(leaky_relu_slope))
              for _ in range(stem_num_cam_bottlenecks)],
            # reshape to ``(b, stem_inner_chans, keys, t)``
            DepthwiseConv2d(stem_inner_chans, stem_out_chans, in_bins, out_bins,
                            kernel_width=1, bias=False),
            torch.nn.BatchNorm2d(stem_out_chans, momentum=bn_momentum),
            get_relu(leaky_relu_slope))
        #
        self.first_stage = self.get_cam_stage(
             stem_out_chans, out_bins, conv1x1,
             cam_num_bottlenecks, cam_hdc_chans, cam_se_bottleneck,
             cam_ksizes, cam_dilations, cam_paddings,
             bn_momentum, leaky_relu_slope, dropout_drop_p)
        #
        self.refiner_stages = torch.nn.ModuleList(
            [self.get_cam_stage(
                stem_out_chans, out_bins, conv1x1,
                refiner_num_bottlenecks, refiner_hdc_chans,
                refiner_se_bottleneck,
                refiner_ksizes, refiner_dilations, refiner_paddings,
                bn_momentum, leaky_relu_slope, dropout_drop_p)
             for _ in range(num_refiner_stages)])

        self.velocity_stage = torch.nn.Sequential(
            SubSpectralNorm(stem_out_chans + 1, out_bins, out_bins, bn_momentum),
            self.get_cam_stage(
                stem_out_chans + 1, out_bins, [out_bins * 2, out_bins],
                num_cam_bottlenecks=1, cam_hdc_chans=4,
                cam_ksizes=((1, 3), (1, 3), (1, 3), (1, 3)),
                cam_dilations=((1, 1), (1, 2), (1, 3), (1, 4)),
                cam_paddings=((0, 1), (0, 2), (0, 3), (0, 4)),
                bn_momentum=bn_momentum, leaky_relu_slope=leaky_relu_slope,
                dropout_p=dropout_drop_p, summary_width=3),
            SubSpectralNorm(1, out_bins, out_bins, bn_momentum))

        # initialize weights
        if init_fn is not None:
            self.apply(lambda module: init_weights(
                module, init_fn, bias_val=0.0))
        self.apply(lambda module: self.set_se_biases(module, se_init_bias))

    @staticmethod
    def set_se_biases(module, bias_val):
        """
        """
        try:
            module.se.set_biases(bias_val)
        except AttributeError:
            pass  # ignore: not a CAM module

    def forward_onsets(self, x):
        """
        """
        xdiff = x.diff(dim=-1)  # (b, melbins, t-1)
        # x+xdiff has shape (b, 2, melbins, t-1)
        x = torch.stack([x[:, :, 1:], xdiff]).permute(1, 0, 2, 3)
        x = self.specnorm(x)
        #
        stem_out = self.stem(x)  # (b, stem_ch, keys, t-1)
        x = self.first_stage(stem_out)  # (b, 1, keys, t-1)
        #
        x_stages = [x]
        for ref in self.refiner_stages:
            ### x = ref(torch.cat([x_stages[-1], stem_out], dim=1))
            x = ref(stem_out) + x_stages[-1]  # residual instead of concat
            x_stages.append(x)
        for st in x_stages:
            st.squeeze_(1)
        #
        return x_stages, stem_out

    def forward(self, x, trainable_onsets=True):
        """
        :param x: Logmel batch of shape ``(b, melbins, t)``
        :returns: List of stage outputs ``(b, keys, t-1)``
        """
        if trainable_onsets:
            x_stages, stem_out = self.forward_onsets(x)
            stem_out = torch.cat([stem_out, x_stages[-1].unsqueeze(1)], dim=1)
        else:
            with torch.no_grad():
                x_stages, stem_out = self.forward_onsets(x)
                stem_out = torch.cat([stem_out, x_stages[-1].unsqueeze(1)],
                                     dim=1)
        #
        velocities = self.velocity_stage(stem_out).squeeze(1)
        #
        return x_stages, velocities


# ##############################################################################
# # ONSET+VELOCITY DECODERS
# ##############################################################################
class OnsetVelocityNmsDecoder(torch.nn.Module):
    """
    Like the Onset NMS decoder, but it also receives a tensor with velocities,
    and
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
    def read_velocities(velmap, batch_idxs, key_idxs, t_idxs, pad_l=0, pad_r=0):
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


def get_ov_demo_model(model_path, num_mels=250, num_keys=88, device="cpu"):
    """
    Returns a function that receives a LogMel of shape ``(mels, t)`` plus a
    threshold, and returns a decoded onset-with-vel pianoroll of shape
    ``(keys, t)``, plus the corresponding onset dataframe.
    """
    model = OnsetVelocity_2(
        in_bins=num_mels, out_bins=num_keys,
        conv1x1=(400, 200, 100),
        leaky_relu_slope=0.1,
        bn_momentum=0.85).to(device)
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
