import torch
import torchio as tio
from pathlib import Path
from mri_segmentation.data import get_data, get_subjects
from mri_segmentation.model import get_model
from mri_segmentation.train import evaluate
from mri_segmentation.metrics import dice_score, hausdorff_score
from mri_segmentation.preprocessing import get_inference_transform
from functools import partial
import json


root = Path('/nmnt/x2-hdd/experiments/pulmonary_trunk/test')
data_dir = root / 'fcd'
labels_path = root / 'targets_fcd_bank.csv'
distmaps_dir = None


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")


iterator_kwargs = {
    'patch_size': 64,
    'samples_per_volume': 8,
    'max_queue_length': 240,
    'training_batch_size': 16,
    'validation_batch_size': 4,
    'num_training_workers': 4,
    'num_validation_workers': 4,
    'patch_overlap': 0
}

n_classes = 6
spacing = (1., 1., 1.)
key = 'patient'


metrics = {
    **{f'dice_{i}': partial(dice_score, n_classes=n_classes, channel=i) for i in range(n_classes)},
    **{f'hausdorff_{i}': partial(hausdorff_score, spacing=spacing, channel=i) for i in range(n_classes)}
}

model_kwargs = {
    'in_channels': 1,
    'out_classes': n_classes,
    'dimensions': 3,
    'num_encoding_blocks': 4,
    'out_channels_first_layer': 16,
    'normalization': 'batch',
    'upsampling_type': 'linear',
    'padding': True,
    'activation': 'PReLU',
    'architecture': 'unet',
    'device': device
}

model = get_model(**model_kwargs)
weights_stem = '6_classes_4_blocks_16_chanels_ce_dice_loss'
models_dir = Path('/nmnt/x2-hdd/experiments/pulmonary_trunk/test/models')
model_path = models_dir / f'model_{weights_stem}.pth'
model.load_state_dict(torch.load(model_path, map_location=device))

names_dict = {
    'with_fcd': root / 'fcd_1_files.json',
    'without_fcd': root / 'fcd_0_files.json'
}

for group, path2names in names_dict.items():
    with open(path2names, 'rt') as f:
        names = json.load(f)

    data_list = get_data(data_dir, labels_path, key, distmaps_dir=distmaps_dir, n_classes=n_classes, names=names)
    subjects = get_subjects(data_list['norm'], data_list['aseg'])
    transform = get_inference_transform()
    data_set = tio.SubjectsDataset(subjects, transform=transform)

    print('Data set:', len(data_set), 'subjects')

    scores = evaluate(model, data_set, metrics, device=device, **iterator_kwargs)

    with open(root / f'fcd_dset_metrics_{group}.json', 'wt') as f:
        json.dump(scores, f)
