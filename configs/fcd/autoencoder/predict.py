import torch
import torchio as tio
from pathlib import Path
from functools import partial
import json

from mri_segmentation.data import get_data, get_subjects, get_sets
from mri_segmentation.model import get_model
from mri_segmentation.train import make_predictions
from mri_segmentation.preprocessing import get_baseline_transforms
from mri_segmentation.metrics import dice_score, hausdorff_score
from mri_segmentation.utils import make_deterministic


root = Path('/nmnt/x2-hdd/experiments/pulmonary_trunk/test')
data_dir = root / 'fcd'
labels_path = root / 'targets_fcd_bank.csv'
distmaps_dir = None


make_deterministic()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
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


with open(root / 'autoencoder' / 'test.json', 'rt') as f:
    test_names = json.load(f)[:2]


with open(root / 'autoencoder' / 'train.json', 'rt') as f:
    train_names = json.load(f)[:100]


transforms = get_baseline_transforms(n_classes)
train_data_list = get_data(data_dir, labels_path, key, distmaps_dir=distmaps_dir, n_classes=n_classes,
                           names=train_names, dropna=False)['aseg'].to_frame().dropna()
test_data_list = get_data(data_dir, labels_path, key, distmaps_dir=distmaps_dir, n_classes=n_classes,
                          names=test_names, dropna=False)['aseg'].to_frame().dropna()
train_subjects = get_subjects(train_data_list['aseg'], train_data_list['aseg'], im_type=tio.LABEL)
test_subjects = get_subjects(test_data_list['aseg'], train_data_list['aseg'], im_type=tio.LABEL)
train_set, val_set, test_set = get_sets(train_subjects, test_subjects, transforms=transforms, train_size=0.95)


print('Training set:', len(train_set), 'subjects')
print('Validation set:', len(val_set), 'subjects')
print('Testing set:', len(test_set), 'subjects')

torch.cuda.empty_cache()

model_kwargs = {
    'in_channels': 6,
    'out_classes': n_classes,
    'dimensions': 3,
    'num_encoding_blocks': 5,
    'out_channels_first_layer': 8,
    'normalization': 'batch',
    'upsampling_type': 'linear',
    'padding': True,
    'activation': 'PReLU',
    'architecture': 'autoencoder',
    'device': device
}

model = get_model(**model_kwargs)

metrics = {
    **{f'dice_{i}': partial(dice_score, n_classes=n_classes, channel=i) for i in range(n_classes)},
    **{f'hausdorff_{i}': partial(hausdorff_score, spacing=spacing, channel=i) for i in range(n_classes)}
}


weights_stem = 'autoencoder_dice'
models_dir = Path('/nmnt/x2-hdd/experiments/pulmonary_trunk/test/models')
model_path = models_dir / f'model_{weights_stem}.pth'

model.load_state_dict(torch.load(model_path, map_location=device))
predictions_path = Path('/nmnt/x2-hdd/experiments/pulmonary_trunk/test/predictions/autoencoder_dice')
make_predictions(model, test_set, predictions_path, **iterator_kwargs)
