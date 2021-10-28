!pip install matplotlib
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import json
import os
from functools import partial
from collections import defaultdict

from scipy.special import softmax as _softmax
from scipy.stats import ttest_ind, kstest

def softmax(x):
    return _softmax(x, axis=0)

def postprocess(predict):
    return np.argmax(predict, axis=0)

def _compute_volumes(mask, n_classes):
    assert len(mask.shape) == 3
    return [(mask == i).sum() for i in range(n_classes)] 
    
import torchio as tio
from mri_segmentation.data import get_data, get_test_data, get_subjects, get_sets
from mri_segmentation.utils import MRI, LABEL
from mri_segmentation.preprocessing import get_baseline_transforms
from mri_segmentation.plotting import plot_central_cuts
from mri_segmentation.utils import prepare_aseg
from sklearn.model_selection import train_test_split
from torchio import DATA

root = Path('C:/Users/Dastan/project')

def get_la5_dset():
    root = Path('C:/Users/Dastan/project')
    data_dir = root / 'la5'
    labels_path = root / 'LA5_actual_targets.csv'

    distmaps_dir = None
    n_classes = 6
    key = 'participant_id'

    data_list = get_data(data_dir, labels_path, key, distmaps_dir=distmaps_dir, n_classes=n_classes)
    subjects = get_subjects(data_list['norm'], data_list['aseg'])
    _, transform = get_baseline_transforms(n_classes)
    data_set = tio.SubjectsDataset(subjects, transform=transform)
    
    return data_set

def get_anat_dset(val=True):
    n_classes = 6
    key = 'Subject'

    data_dir = root / 'test'
    train_data_dir = data_dir / 'fs_segmentation'
    test_data_dir = data_dir / 'anat'
    labels_path = data_dir / 'unrestricted_hcp_freesurfer.csv'
    distmaps_dir = None

    _, transform = get_baseline_transforms(n_classes)
    train_data_list = get_data(train_data_dir, labels_path, key, distmaps_dir=distmaps_dir, n_classes=n_classes)
    test_data_list = get_test_data(test_data_dir, key)
    train_subjects = get_subjects(train_data_list['norm'], train_data_list['aseg'])
    test_subjects = get_subjects(test_data_list['norm'], test_data_list['aseg'])
    train_set, val_set, test_set = get_sets(train_subjects, test_subjects, transforms=(transform, transform))
    
    return val_set if val else train_set
    
def compute_volumes(data_set, predictions_dir, n_classes=6):
    target_volumes, pred_volumes, target_uids, pred_uids = [], [], [], []

    for sample in tqdm(data_set):
        target_mask = torch.argmax(sample[LABEL][DATA], dim=0)
        uid = sample[MRI]['path'].split('/')[-1][:-7]
        
        if f'{uid}' in list(map(str, map(lambda x: x.stem, predictions_dir.iterdir()))):
            pred_mask = postprocess(np.load(predictions_dir / f'{uid}.npy'))
            pred_volumes.append(_compute_volumes(pred_mask, n_classes))
            pred_uids.append(uid)

        target_volumes.append(_compute_volumes(target_mask, n_classes))
        target_uids.append(uid)


    columns = ['OTHER', 'VENTRCL', 'BRN_STEM', 'HIPPOCMPS', 'AMYGDL', 'GM']
    df_target = pd.DataFrame(np.array(target_volumes), target_uids, columns)
    df_pred = pd.DataFrame(np.array(pred_volumes), pred_uids, columns)
    
    return df_pred, df_target
    
def _format(s):
    s = s[:-5]
    if s.startswith('LA5_freesurfer_'):
        s = s[15:]
    else:
        raise ValueError(f'Failed to parse string "{s}"')
    
    return s
    
root = Path('C:/Users/Dastan/project')
train_set_anat = get_anat_dset(False)

def _compute_volumes(mask, n_classes):
    assert len(mask.shape) == 3
    return [(mask == i).sum().item() for i in range(n_classes)]

for sample in tqdm(train_set_anat):
    target_mask = sample[LABEL][DATA].argmax(dim=0)
    uid = sample[MRI]['path'].split('/')[-1][:-7]
    with open(root / 'anat_volumes' / f'{uid}.json', 'wt') as f:
        json.dump(_compute_volumes(target_mask, 6), f)
        
#Both colab and local jupyter notebook stop there due to lack of memory.        
