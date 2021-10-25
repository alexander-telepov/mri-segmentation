import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from mri_segmentation.data import get_data, get_test_data, get_subjects, get_sets
from mri_segmentation.model import get_model
from mri_segmentation.train import make_predictions
from mri_segmentation.preprocessing import get_baseline_transforms


data_dir = Path('/nmnt/x2-hdd/experiments/pulmonary_trunk/test/anat-20210925T153621Z-001/')
train_data_dir = data_dir / 'anat' / 'fs_segmentation'
test_data_dir = data_dir / 'test'
labels_path = data_dir / 'anat' / 'unrestricted_hcp_freesurfer.csv'
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
key = 'Subject'


train_data_list = get_data(train_data_dir, labels_path, key, distmaps_dir=distmaps_dir, n_classes=n_classes)
test_data_list = get_test_data(test_data_dir, key)
train_subjects = get_subjects(train_data_list['norm'], train_data_list['aseg'])
test_subjects = get_subjects(test_data_list['norm'], test_data_list['aseg'])
training_subjects, validation_subjects = train_test_split(
    train_subjects, train_size=0.9, shuffle=True, random_state=42
)
transforms = get_baseline_transforms(n_classes)
train_set, val_set, test_set = get_sets(train_subjects, test_subjects, transforms=transforms)

print('Training set:', len(train_set), 'subjects')
print('Validation set:', len(val_set), 'subjects')
print('Testing set:', len(test_set), 'subjects')

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
model.load_state_dict(torch.load(model_path, map_location=device))

# no sense to predict on train, because it may overfit
# predictions_path_train = Path('/nmnt/x2-hdd/experiments/pulmonary_trunk/test/predictions/anat_train')
# make_predictions(model, train_set, predictions_path_train, **iterator_kwargs)

predictions_path_val = Path('/nmnt/x2-hdd/experiments/pulmonary_trunk/test/predictions/anat_val')
make_predictions(model, val_set, predictions_path_val, **iterator_kwargs)

predictions_path_val = Path('/nmnt/x2-hdd/experiments/pulmonary_trunk/test/predictions/anat_test')
make_predictions(model, test_set, predictions_path_val, **iterator_kwargs)
