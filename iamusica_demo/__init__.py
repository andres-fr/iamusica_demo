# !/usr/bin/env python
# -*- coding:utf-8 -*-


"""
"""


import os
#
import torch
#
from . import __path__ as ROOT_PATH


# ##############################################################################
# # PATHING
# ##############################################################################
ASSETS_PATH = os.path.join(ROOT_PATH[0], "assets")
OV_MODEL_PATH = os.path.join(
    ASSETS_PATH,
    "OnsetsAndVelocities_" +
    "2023_03_04_09_53_53.289step=43500_f1=0.9675__0.9480.torch")
OV_MODEL_CONV1X1_HEAD = [200, 200]
OV_MODEL_LRELU_SLOPE = 0.1

# ##############################################################################
# # PIPELINE STATIC PROPERTIES
# ##############################################################################
WAV_SAMPLERATE = 16000
MEL_FRAME_SIZE = 2048
MEL_FRAME_HOP = 384
NUM_MELS = 229
NUM_PIANO_KEYS = 88
MEL_FMIN, MEL_FMAX = (50, 8000)
MEL_WINDOW = torch.hann_window
