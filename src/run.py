from comet_ml import Experiment
import torch
from torch.cuda.amp import GradScaler
import numpy as np
from pathlib import Path
from functools import partial
from surface_distance import metrics
from .data import get_train_data, get_test_data, get_subjects, get_sets, get_loaders
from .model import get_model
from .train import train, evaluate
from .loss import DiceLoss, BoundaryLoss


experiment = Experiment(
    api_key="HVUHHBi3VVGCPG3tcvpRsZUex",
    project_name="neuro_hw2",
    workspace="alexander-telepov",
)

data_dir = Path('/gpfs/gpfs0/a.telepov/mri')
train_data_dir = data_dir / 'anat' / 'fs_segmentation'
test_data_dir = data_dir / 'test'
labels_dir = data_dir / 'anat'
distmaps_dir = data_dir / 'anat' / 'dist_maps'


determesitic = True
baseline_transforms = False

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
    'patch_size': 96,
    'samples_per_volume': 8,
    'max_queue_length': 240,
    'training_batch_size': 10,
    'validation_batch_size': 4,
    'num_training_workers': 4,
    'num_validation_workers': 4,
    'patch_overlap': 0
}

n_classes = 6
spacing = (1., 1., 1.)
num_epochs = 20

train_data_list = get_train_data(train_data_dir, labels_dir, distmaps_dir, n_classes=n_classes)
test_data_list = get_test_data(test_data_dir)
train_subjects = get_subjects(train_data_list['norm'], train_data_list['aseg'],
                              [train_data_list[f'distmap_{i}'] for i in range(n_classes)])
test_subjects = get_subjects(test_data_list['norm'], test_data_list['aseg'])
train_set, val_set, test_set = get_sets(train_subjects, test_subjects, baseline_transforms=baseline_transforms)
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
    'architecture': 'attention_unet'
}

model, optimizer, scheduler = get_model(device, **model_kwargs)
optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=0.01)

dice_loss = DiceLoss(n_classes)
boundary_loss = BoundaryLoss(idc=list(range(n_classes)))


def dice_loss(logits, targets, *args, **kwargs):
    return dice_loss(logits, targets.squeeze(1), softmax=True).mean()[0]


def boundary_loss(logits, targets, distmaps, *args, alpha=0.01, **kwargs):
    return alpha * boundary_loss(torch.softmax(logits, dim=1), distmaps)


criterions = {'dice': dice_loss, 'boundary_loss': boundary_loss}


def dice_score(prediction, targets):
    return dice_loss(prediction, targets, softmax=True)[1]


def hausdorff_scores(prediction, targets, spacing=spacing, channel=0, level=95):
    distances = metrics.compute_surface_distances((targets.squeeze(0).numpy() == channel),
                                                  (np.argmax(prediction.squeeze(0).numpy(), axis=0) == channel), spacing)
    return metrics.compute_robust_hausdorff(distances, level)


metrics = {
    'dice_score': dice_score,
    **{f'hausdorff_{i}': partial(hausdorff_scores, channel=i) for i in range(n_classes)}
}


experiment.set_name("20 epochs, 4 encoding blocks, 8 out, 96 patch, extra transforms, boundary loss, attention")
weights_stem = 'e20_bl4_o8_ch8_p96_t-extra_l-boundary_attention'
models_dir = Path('/gpfs/gpfs0/a.telepov/mri/models')
model_path = models_dir / f'model_{weights_stem}.pth'

scaler = GradScaler()
train(experiment, num_epochs, train_loader, val_set, model, optimizer, scheduler, criterions, metrics,
      save_path=model_path, scaler=scaler)


model.load_state_dict(model_path, map_location=device)
test_scores = evaluate(model, test_set, **iterator_kwargs)
print(test_scores)
print(f"\nTesting mean score: DICE {np.mean(test_scores['dice']):0.3f}")

experiment.log_metric("avg_test_dice", np.mean(test_scores['dice']))
for i, subject in enumerate(test_subjects):
    experiment.log_metric(f"test_subj_{subject}_dice", np.mean(test_scores['dice'][i]))
