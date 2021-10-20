import torch
import torchio
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms as torch_transforms
from scipy.ndimage import distance_transform_edt as eucl_distance
import os
from tqdm import tqdm
from functools import partial
from operator import itemgetter
from .utils import sset, one_hot, uniq, MRI, LABEL, DIST_MAP
from .preprocessing import get_transforms


def get_data(data_dir, labels_path, key, distmaps_dir=None, n_classes=1):
    columns = [key, 'norm', 'aseg']
    if distmaps_dir:
        columns.extend([f'distmap_{i}' for i in range(n_classes)])

    data_list = pd.DataFrame(columns=columns)
    labels = pd.read_csv(labels_path)
    data_list[key] = labels[key]

    for i in tqdm(os.listdir(data_dir)):
        for j in range(len(data_list[key])):
            if str(data_list[key].iloc[j]) in i:
                if 'norm' in i:  # copying path to the column norm
                    data_list['norm'].iloc[j] = str(data_dir / i)
                elif 'aseg' in i:  # copying path to second column
                    data_list['aseg'].iloc[j] = str(data_dir / i)

    if distmaps_dir:
        for i in tqdm(os.listdir(distmaps_dir)):
            for j in range(len(data_list[key])):
                if str(data_list[key].iloc[j]) in i:
                    k = i[-8]
                    data_list[f'distmap_{k}'].iloc[j] = str(distmaps_dir / i)

    data_list.dropna(inplace=True)
    return data_list


def get_test_data(data_dir, key):
    test_subjects = [100206, 100307, 100408]

    testing_data_list = pd.DataFrame({
        key: test_subjects,
        'norm': [f'{data_dir}/HCP_T1_fs6_{subject}_norm.nii.gz' for subject in test_subjects],
        'aseg': [f'{data_dir}/HCP_T1_fs6_{subject}_aparc+aseg.nii.gz' for subject in test_subjects]
    })

    return testing_data_list


def get_subjects(inputs, targets, distmaps=None, n_classes=1):
    """
    The function creates list of torchio.subject from the list of files from customized dataloader.
    """
    subjects = []
    for i, (image_path, label_path) in enumerate(zip(inputs, targets)):
        subject_dict = {
            MRI: torchio.Image(image_path, torchio.INTENSITY),
            LABEL: torchio.Image(label_path, torchio.LABEL),
        }

        if distmaps is not None:
            for k in range(n_classes):
                subject_dict[DIST_MAP + f'_{k}'] = torchio.Image(distmaps[k].iloc[i], torchio.LABEL)

        subject = torchio.Subject(subject_dict)
        subjects.append(subject)

    return subjects


def get_sets(subjects, testing_subjects, baseline_transforms=True, train_size=0.9):

    train_transform, validation_transform = get_transforms(baseline_transforms)

    training_subjects, validation_subjects = train_test_split(
        subjects, train_size=train_size, shuffle=True, random_state=42
    )

    training_set = torchio.SubjectsDataset(
        training_subjects, transform=train_transform)

    validation_set = torchio.SubjectsDataset(
        validation_subjects, transform=validation_transform)

    testing_set = torchio.SubjectsDataset(
        testing_subjects, transform=validation_transform)

    return training_set, validation_set, testing_set


def get_loaders(training_set, validation_set, max_queue_length=10, samples_per_volume=2, patch_size=20,
                training_batch_size=2, validation_batch_size=2, num_training_workers=2, num_validation_workers=2,
                **kwargs):
    patches_training_set = torchio.Queue(
        subjects_dataset=training_set,
        max_length=max_queue_length,
        samples_per_volume=samples_per_volume,
        sampler=torchio.sampler.UniformSampler(patch_size),
        num_workers=num_training_workers,
        shuffle_subjects=True,
        shuffle_patches=True,
    )

    patches_validation_set = torchio.Queue(
        subjects_dataset=validation_set,
        max_length=max_queue_length,
        samples_per_volume=samples_per_volume,
        sampler=torchio.sampler.UniformSampler(patch_size),
        num_workers=num_validation_workers,
        shuffle_subjects=False,
        shuffle_patches=False,
    )

    training_loader = torch.utils.data.DataLoader(
        patches_training_set, batch_size=training_batch_size)

    validation_loader = torch.utils.data.DataLoader(
        patches_validation_set, batch_size=validation_batch_size, shuffle=False)

    return training_loader, validation_loader


def class2one_hot(seg, K):
    # Breaking change but otherwise can't deal with both 2d and 3d
    # if len(seg.shape) == 3:  # Only w, h, d, used by the dataloader
    #     return class2one_hot(seg.unsqueeze(dim=0), K)[0]

    assert sset(seg, list(range(K))), (uniq(seg), K)

    b, *img_shape = seg.shape  # type: Tuple[int, ...]

    device = seg.device
    res = torch.zeros((b, K, *img_shape), dtype=torch.int32, device=device).scatter_(1, seg[:, None, ...], 1)

    assert res.shape == (b, K, *img_shape)
    assert one_hot(res)

    return res


def gt_transform(resolution, K):
    return torch_transforms.Compose([
        lambda img: np.array(img)[...],
        lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
        partial(class2one_hot, K=K),
        itemgetter(0)  # Then pop the element to go back to img shape
    ])


def one_hot2dist(seg: np.ndarray, resolution=None, dtype=None):
    assert one_hot(torch.tensor(seg), axis=0)
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[k] = eucl_distance(negmask, sampling=resolution) * negmask \
                - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel

    return res


def dist_map_transform(resolution, K):
    return torch_transforms.Compose([
        gt_transform(resolution, K),
        lambda t: t.cpu().numpy(),
        partial(one_hot2dist, resolution=resolution),
        lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])
