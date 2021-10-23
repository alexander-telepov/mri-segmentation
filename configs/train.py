from comet_ml import Experiment
import torch
from torch.cuda.amp import GradScaler
import numpy as np
from pathlib import Path
from functools import partial
from surface_distance import metrics as _metrics
from mri_segmentation.data import get_data, get_test_data, get_subjects, get_sets, get_loaders
from mri_segmentation.model import get_model
from mri_segmentation.train import train, evaluate
from mri_segmentation.loss import DiceLoss, BoundaryLoss
from mri_segmentation.preprocessing import get_baseline_transforms


experiment = Experiment(
    api_key="HVUHHBi3VVGCPG3tcvpRsZUex",
    project_name="neuro_project",
    workspace="alexander-telepov",
)

data_dir = Path('/nmnt/x2-hdd/experiments/pulmonary_trunk/test/anat-20210925T153621Z-001/')
train_data_dir = data_dir / 'anat' / 'fs_segmentation'
test_data_dir = data_dir / 'test'
labels_path = data_dir / 'anat' / 'unrestricted_hcp_freesurfer.csv'
distmaps_dir = None


determesitic = True

if determesitic:
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
num_epochs = 10
key = 'Subject'

transforms = get_baseline_transforms()
train_data_list = get_data(train_data_dir, labels_path, key, distmaps_dir=distmaps_dir, n_classes=n_classes)
test_data_list = get_test_data(test_data_dir, key)
train_subjects = get_subjects(train_data_list['norm'], train_data_list['aseg'])
test_subjects = get_subjects(test_data_list['norm'], test_data_list['aseg'])
train_set, val_set, test_set = get_sets(train_subjects, test_subjects, transforms=transforms)
train_loader, val_loader = get_loaders(train_set, val_set, **iterator_kwargs)

print('Training set:', len(train_set), 'subjects')
print('Validation set:', len(val_set), 'subjects')
print('Testing set:', len(test_set), 'subjects')

torch.cuda.empty_cache()

model_kwargs = {
    'in_channels': 1,
    'out_classes': n_classes,
    'dimensions': 3,
    'num_encoding_blocks': 4,
    'out_channels_first_layer': 8,
    'normalization': 'batch',
    'upsampling_type': 'linear',
    'padding': True,
    'activation': 'PReLU',
    'architecture': 'unet',
    'device': device
}

model = get_model(**model_kwargs)
optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=0.01)

_dice_loss = DiceLoss(n_classes)
_boundary_loss = BoundaryLoss(idc=list(range(n_classes)))


def dice_loss(logits, targets, *args, **kwargs):
    return _dice_loss(logits, targets.squeeze(1), softmax=True)


def boundary_loss(logits, targets, distmaps, *args, alpha=0.01, **kwargs):
    return alpha * _boundary_loss(torch.softmax(logits, dim=1), distmaps)


criterions = {'dice': dice_loss}


def dice_score(prediction, targets, channel=0):
    prediction = torch.softmax(prediction, dim=1)
    targets = _dice_loss.one_hot_encoder(targets)

    assert prediction.size() == targets.size(), \
        'predict {} & target {} shape do not match'.format(prediction.size(), targets.size())

    dice = _dice_loss.dice_loss(prediction[:, channel], targets[:, channel])

    return 1.0 - dice.item()


def hausdorff_score(prediction, targets, spacing=spacing, channel=0, level=95):
    distances = _metrics.compute_surface_distances((targets.squeeze(0).numpy() == channel),
                                                   (np.argmax(prediction.squeeze(0).numpy(), axis=0) == channel),
                                                   spacing)
    return _metrics.compute_robust_hausdorff(distances, level)


metrics = {
    **{f'dice_{i}': partial(dice_score, channel=i) for i in range(n_classes)},
    **{f'hausdorff_{i}': partial(hausdorff_score, channel=i) for i in range(n_classes)}
}


experiment.set_name("10 epochs, 4 encoding blocks, 8 out, 64 patch, baseline transforms")
weights_stem = 'e10_bl4_o8_p64_t-base_m-unet'
models_dir = Path('/nmnt/x2-hdd/experiments/pulmonary_trunk/test/models')
model_path = models_dir / f'model_{weights_stem}.pth'

scaler = GradScaler()
train(experiment, num_epochs, train_loader, val_set, model, optimizer, criterions, metrics, scheduler=scheduler,
      save_path=model_path, scaler=scaler)


model.load_state_dict(torch.load(model_path, map_location=device))
test_scores = evaluate(model, test_set, metrics, **iterator_kwargs)
print(test_scores)
print(f"\nTesting mean score: DICE {np.mean(test_scores['dice']):0.3f}")

experiment.log_metric("avg_test_dice", np.mean(test_scores['dice']))
for i, subject in enumerate(test_subjects):
    experiment.log_metric(f"test_subj_{subject}_dice", test_scores['dice'][i])
