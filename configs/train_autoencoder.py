from comet_ml import Experiment
import torch
import torchio as tio
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import GradScaler
import numpy as np
from pathlib import Path
from functools import partial
import json

from mri_segmentation.data import get_data, get_subjects, get_sets, get_loaders
from mri_segmentation.model import get_model
from mri_segmentation.train import train, evaluate
from mri_segmentation.preprocessing import get_baseline_transforms
from mri_segmentation.metrics import dice_score, hausdorff_score
from mri_segmentation.utils import make_deterministic, MRI


experiment = Experiment(
    api_key="HVUHHBi3VVGCPG3tcvpRsZUex",
    project_name="neuro_project",
    workspace="alexander-telepov",
)

root = Path('/nmnt/x2-hdd/experiments/pulmonary_trunk/test')
# TODO: to use predictions we need to store it in nifty format
data_dir = root / 'fcd'
labels_path = root / 'targets_fcd_bank.csv'
distmaps_dir = None


make_deterministic()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


iterator_kwargs = {
    # TODO: patch should be bigger
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
num_epochs = 5
key = 'patient'

with open(root / 'autoencoder' / 'test.json', 'rt') as f:
    test_names = json.load(f)


with open(root / 'autoencoder' / 'train.json', 'rt') as f:
    train_names = json.load(f)


transforms = get_baseline_transforms(n_classes)
train_data_list = get_data(data_dir, labels_path, key, distmaps_dir=distmaps_dir, n_classes=n_classes,
                           names=train_names, dropna=False)['aseg'].to_frame().dropna()
test_data_list = get_data(data_dir, labels_path, key, distmaps_dir=distmaps_dir, n_classes=n_classes,
                          names=test_names, dropna=False)['aseg'].to_frame().dropna()
train_subjects = get_subjects(train_data_list['aseg'], train_data_list['aseg'], im_type=tio.LABEL)
test_subjects = get_subjects(test_data_list['aseg'], train_data_list['aseg'], im_type=tio.LABEL)
train_set, val_set, test_set = get_sets(train_subjects, test_subjects, transforms=transforms, train_size=0.95)
train_loader, val_loader = get_loaders(train_set, val_set, **iterator_kwargs)

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
optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=0.01)


ce_loss = CrossEntropyLoss()


def cross_entropy(logits, targets):
    return ce_loss(logits, targets['segm_mask'].argmax(dim=1))


criterions = {'cross_entropy': cross_entropy}


metrics = {
    **{f'dice_{i}': partial(dice_score, n_classes=n_classes, channel=i) for i in range(n_classes)},
    **{f'hausdorff_{i}': partial(hausdorff_score, spacing=spacing, channel=i) for i in range(n_classes)}
}


experiment.set_name("train autoencoder")
weights_stem = 'autoencoder'
models_dir = Path('/nmnt/x2-hdd/experiments/pulmonary_trunk/test/models')
model_path = models_dir / f'model_{weights_stem}.pth'

scaler = GradScaler()
train(experiment, num_epochs, train_loader, val_set, model, optimizer, criterions, metrics, scheduler=scheduler,
      save_path=model_path, scaler=scaler, device=device)


model.load_state_dict(torch.load(model_path, map_location=device))
test_scores = evaluate(model, test_set, metrics, **iterator_kwargs)
print(test_scores)
print(f"\nTesting mean score: DICE {np.mean(test_scores['dice']):0.3f}")

experiment.log_metric("avg_test_dice", np.mean(test_scores['dice']))
for i, subject in enumerate(test_subjects):
    experiment.log_metric(f"test_subj_{subject}_dice", test_scores['dice'][i])
