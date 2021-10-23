import torch
import torchio as tio
from pathlib import Path
from mri_segmentation.data import get_data, get_subjects
from mri_segmentation.model import get_model
from mri_segmentation.train import make_predictions


data_dir = Path('/nmnt/x2-hdd/experiments/pulmonary_trunk/test/fcd')
labels_path = Path('/nmnt/x2-hdd/experiments/pulmonary_trunk/test/targets_fcd_bank.csv')
distmaps_dir = None


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
key = 'patient'

data_list = get_data(data_dir, labels_path, key, distmaps_dir=distmaps_dir, n_classes=n_classes)
subjects = get_subjects(data_list['norm'], data_list['aseg'])
transform = tio.Compose([
    tio.ToCanonical(),
    tio.Resample((1., 1., 1.))
])
data_set = tio.SubjectsDataset(subjects, transform=transform)

print('Data set:', len(data_set), 'subjects')

torch.cuda.empty_cache()

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

predictions_path = Path('/nmnt/x2-hdd/experiments/pulmonary_trunk/test/predictions/fcd')

model.load_state_dict(torch.load(model_path, map_location=device))
make_predictions(model, data_set, predictions_path, **iterator_kwargs)
