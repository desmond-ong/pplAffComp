from __future__ import division, print_function, absolute_import

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


from pyro.distributions.util import broadcast_shape



'''
This file contains helpful functions and variables
'''

OUTCOME_VAR_NAMES = ["payoff1", "payoff2", "payoff3", "prob1", "prob2", "prob3", "win", "winProb", "angleProp"]
EMOTION_VAR_NAMES = ["happy", "sad", "anger", "surprise", "disgust", "fear", "content", "disapp"]
OUTCOME_VAR_DIM = len(OUTCOME_VAR_NAMES)
EMOTION_VAR_DIM = len(EMOTION_VAR_NAMES)

FACE_FILENAMES = ["face_-3_4_1", "face_-2_3_1", "face_-2_1_1", \
    "face_-3_0_1", "face_-1_2_1", "face_0_0_1", "face_0_4_1", \
    "face_1_2_1", "face_3_0_1", "face_2_1_1", "face_2_3_1", \
    "face_3_4_1", "aface_1", "aface_2", "aface_3", \
    "aface_4", "aface_5", "aface_6"]



def ToTensor(img):
    """Convert ndarrays in sample to Tensors."""

    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    newimg = img.transpose((2, 0, 1))
    return torch.from_numpy(newimg)

def TensorToPILImage(t):
    """Convert Tensors in sample to ndarrays."""

    # swap color axis because
    # torch image: C X H X W
    # numpy image: H x W x C
    t1 = np.array(t, dtype='uint8')
    newimg = t1.transpose((1, 2, 0))
    return newimg


class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""
    def forward(self, x):
        return x * F.sigmoid(x)


def swish(x):
    return x * F.sigmoid(x)


