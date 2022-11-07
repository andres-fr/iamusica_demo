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
    "OnsetVelocity_2_2022_09_08_01_27_40.139step=95000_f1=0.9640.torch")

# ##############################################################################
# # PIPELINE STATIC PROPERTIES
# ##############################################################################
WAV_SAMPLERATE = 16000
MEL_FRAME_SIZE = 2048
MEL_FRAME_HOP = 384
NUM_MELS = 250
NUM_PIANO_KEYS = 88
MEL_FMIN, MEL_FMAX = (50, 8000)
MEL_WINDOW = torch.hann_window
