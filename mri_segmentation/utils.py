import torch
import numpy as np
from torchio import DATA
from typing import cast
from torch import Tensor


MRI = 'MRI'
LABEL = 'SEGM_MASK'
DIST_MAP = 'DIST_MAP'

CHANNELS_DIMENSION = 6
SPATIAL_DIMENSIONS = 2, 3, 4

VENTRCL = [4, 5, 15, 43, 44, 72]
BRN_STEM = [16]
HIPPOCMPS = [17, 53]
AMYGDL = [18, 54]
GM = [
    1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013,
    1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024,
    1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035,
    2000, 2001, 2002, 2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
    2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022,
    2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033,
    2034, 2035
]

LABELS = VENTRCL + BRN_STEM + HIPPOCMPS + AMYGDL + GM


def prepare_aseg(targets):
    """
    The function binarize the data with the LABEL list.
   """
    targets = targets.to(torch.int64)
    targets = np.where(np.isin(targets, LABELS, invert=True), 0, targets)
    targets = np.where(np.isin(targets, VENTRCL), 1, targets)
    targets = np.where(np.isin(targets, BRN_STEM), 2, targets)
    targets = np.where(np.isin(targets, HIPPOCMPS), 3, targets)
    targets = np.where(np.isin(targets, AMYGDL), 4, targets)
    targets = np.where(np.isin(targets, GM), 5, targets)

    return torch.tensor(targets).to(torch.int64)


def prepare_batch(batch, device):
    """
    The function loading *nii.gz files, sending to the devise.
    For the LABEL it binarize the data.
    """
    inputs = batch[MRI][DATA].to(device)
    if LABEL in batch.keys():
        segm_mask = batch[LABEL][DATA]
        segm_mask = segm_mask.to(device)
    else:
        segm_mask = None

    dist_maps_keys = [key for key in batch.keys() if DIST_MAP in key]
    if dist_maps_keys:
        dist_maps = [batch[DIST_MAP + f'_{i}'][DATA] for i in range(len(dist_maps_keys))]
        dist_maps = torch.cat(dist_maps, dim=1).to(device)
    else:
        dist_maps = None

    targets = {
        'segm_mask': segm_mask,
        'dist_maps': dist_maps
    }

    return inputs, targets


def simplex(t, axis=1):
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def uniq(a):
    return set(torch.unique(a.cpu()).numpy())


def sset(a, sub):
    return uniq(a).issubset(sub)


def one_hot(t, axis=1):
    return simplex(t, axis) and sset(t, [0, 1])


def make_deterministic():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
