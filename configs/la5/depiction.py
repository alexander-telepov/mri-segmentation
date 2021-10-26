!pip install --quiet --upgrade comet_ml
from comet_ml import Experiment
!pip install --quiet --upgrade unet 
!pip install --quiet --upgrade nibabel 
import torch
import numpy as np
from unet import UNet
from google.colab import drive
drive.mount('/content/drive')
import sys
from pathlib import Path
sys.path.insert(0, '/content/drive/My Drive/neuroimage_final_project/path')

model_path = Path('/content/drive/My Drive/neuroimage_final_project/model_6_classes_4_blocks_16_chanels_ce_dice_loss.pth')
device = None
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    
experiment = Experiment(
    api_key="1oyzHrJhjofZ4uu1imMYi4y9Y",
    project_name="project-2",
    workspace="d45t4n",
)

CHANNELS_DIMENSION = 6
SPATIAL_DIMENSIONS = 2, 3, 4

VENTRCL =  [4,5,15,43,44,72]# 1
BRN_STEM = [16] # 2
HIPPOCMPS = [17, 53] # 3
AMYGDL = [18, 54] # 4
GM = [1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013,
       1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024,
       1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035,
       2000, 2001, 2002, 2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
       2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022,
       2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033,
       2034, 2035] # 5

LABELS = VENTRCL + BRN_STEM + HIPPOCMPS + AMYGDL + GM # all of interest


def prepare_aseg(targets):
    """
    The function binarises the data  with the LABEL list.
   """
    targets = np.where(np.isin(targets, LABELS, invert = True), 0, targets)
    targets = np.where(np.isin(targets, VENTRCL), 1, targets)
    targets = np.where(np.isin(targets, BRN_STEM), 2, targets)
    targets = np.where(np.isin(targets, HIPPOCMPS), 3, targets)
    targets = np.where(np.isin(targets, AMYGDL), 4, targets)
    targets = np.where(np.isin(targets, GM), 5, targets)


    return targets
  
def get_model_and_optimizer(device, num_encoding_blocks=4, out_channels_first_layer=8, patience=3):
    #Better to train with num_encoding_blocks >=3, out_channels_first_layer>=4 '''
    #repoducibility
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
      
    model = UNet(
          in_channels=1,
          out_classes=6,
          dimensions=3,
          num_encoding_blocks=num_encoding_blocks,
          out_channels_first_layer=out_channels_first_layer,
          normalization='batch',
          upsampling_type='linear',
          padding=True,
          activation='PReLU',
      ).to(device)
      
    optimizer = torch.optim.AdamW(model.parameters())
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, threshold=0.01)
    
    return model, optimizer, scheduler

model, optimizer, scheduler = get_model_and_optimizer(device, out_channels_first_layer=16)


import matplotlib.pyplot as plt
import torch
import numpy as np
import nibabel

def plot_central_cuts(img, title=""):
    """
    param image: tensor or np array of shape (CxDxHxW) if t is None
    """
    if isinstance(img, torch.Tensor):
        img = img.numpy()
        if (len(img.shape) > 3):
            img = img[0,:,:,:]
                
    elif isinstance(img, nibabel.nifti1.Nifti1Image):    
        img = img.get_fdata()
   
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(3 * 6, 6))
    axes[0].imshow(img[ img.shape[0] // 2, :, :])
    axes[1].imshow(img[ :, img.shape[1] // 2, :])
    axes[2].imshow(img[ :, :, img.shape[2] // 2])
    
    plt.show()
    
def plot_predicted(img, seg, delta = 0, title=""):
    """
    param image: tensor or np array of shape (CxDxHxW) if t is None
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
        if (len(img.shape) == 5):
            img = img[0,0,:,:,:]
        elif (len(img.shape) == 4):
            img = img[0,:,:,:]
                
    elif isinstance(img, nibabel.nifti1.Nifti1Image):    
        img = img.get_fdata()
        
    if isinstance(seg, torch.Tensor):
        seg= seg[0].cpu().numpy().astype(np.uint8)
   
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(3 * 4, 4))
    axes[0].imshow(img[ img.shape[0] // 2 + delta, :, :])
    axes[1].imshow(seg[ seg.shape[0] // 2 + delta, :, :])
    intersect = img[ img.shape[0] // 2 + delta, :, :] + seg[ seg.shape[0] // 2 + delta, :, :]*100
    axes[2].imshow(intersect, cmap='gray')
    
    plt.show()
    
import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np
import nibabel

def plot_central_cuts(img, title=""):
    """
    param image: tensor or np array of shape (CxDxHxW) if t is None
    """
    if isinstance(img, torch.Tensor):
        img = img.numpy()
        if (len(img.shape) > 3):
            img = img[0,:,:,:]
                
    elif isinstance(img, nibabel.nifti1.Nifti1Image):    
        img = img.get_fdata()
   
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(3 * 4, 4))
    axes[0].imshow(img[ img.shape[0] // 2, :, :])
    axes[1].imshow(img[ :, img.shape[1] // 2, :])
    axes[2].imshow(img[ :, :, img.shape[2] // 2])
    
    plt.show()
    
def plot_predicted(img, seg, delta = 0, title=""):
    """
    param image: tensor or np array of shape (CxDxHxW) if t is None
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
        if (len(img.shape) == 5):
            img = img[0,0,:,:,:]
        elif (len(img.shape) == 4):
            img = img[0,:,:,:]
                
    elif isinstance(img, nibabel.nifti1.Nifti1Image):    
        img = img.get_fdata()
        
    if isinstance(seg, torch.Tensor):
        seg= seg[0].cpu().numpy().astype(np.uint8)
   
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(3 * 4, 4))
    axes[0].imshow(img[ img.shape[0] // 2 + delta, :, :])
    axes[1].imshow(seg[ seg.shape[0] // 2 + delta, :, :])
    intersect = img[ img.shape[0] // 2 + delta, :, :] + seg[ seg.shape[0] // 2 + delta, :, :]*100
    axes[2].imshow(intersect, cmap='gray')
    
    plt.show()
    
import pandas as pd
!pip install --quiet --upgrade torchio
import torchio 
import enum
from torchio import AFFINE, DATA, PATH, TYPE, STEM

"""
    Code adapted from: https://github.com/fepegar/torchio#credits

        Credit: Pérez-García et al., 2020, TorchIO: 
        a Python library for efficient loading, preprocessing, 
        augmentation and patch-based sampling of medical images in deep learning.

"""

MRI = 'MRI'
LABEL = 'LABEL'

class Action(enum.Enum):
    TRAIN = 'Training'
    VALIDATE = 'Validation'

def get_torchio_dataset(inputs, targets, transform):
    """
    The function creates dataset from the list of files from cunstumised dataloader.
    """
    subjects = []
    for (image_path, label_path) in zip(inputs, targets):
        subject_dict = {
            MRI : torchio.Image(image_path, torchio.INTENSITY),
            LABEL: torchio.Image(label_path, torchio.LABEL),
        }
        subject = torchio.Subject(subject_dict)
        subjects.append(subject)
    
    if transform:
        dataset = torchio.SubjectsDataset(subjects, transform = transform)
    elif not transform:
        dataset = torchio.SubjectsDataset(subjects)
    
    return dataset, subjects
  
root = Path('/content/drive/My Drive/neuroimage_final_project')
test_norm_dir = root / 'la5' / 'vis'
test_subjects = [10159, 10171, 10189]

testing_data_list = pd.DataFrame({
    'Subject': test_subjects,
    'norm': [f'{test_norm_dir}/LA5_freesurfer_sub-{subject}_norm.nii.gz' for subject in test_subjects],
    'aseg': [f'{test_norm_dir}/LA5_freesurfer_sub-{subject}_aparc+aseg.nii.gz' for subject in test_subjects]
})

testing_data, testing_subjects = get_torchio_dataset(testing_data_list['norm'], testing_data_list['aseg'], False)
testing_set = torchio.SubjectsDataset(testing_subjects)
testing_data_list.head()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

model.load_state_dict(torch.load(model_path, map_location='cpu'))

!git clone https://github.com/deepmind/surface-distance.git
!pip install surface-distance/
from surface_distance import metrics

def validate(label, pred):
    ''' Computes DICE and Hausdorf95 measures
    '''

    test_res = pd.DataFrame(columns = ['DICE_1', 'DICE_2', 'DICE_3', 'DICE_4', 'DICE_5',
                                      'Hausdorff95_1', 'Hausdorff95_2', 'Hausdorff95_3', 'Hausdorff95_4', 'Hausdorff95_5'])
    # class 1
    distances = metrics.compute_surface_distances((label[0] == 1), (pred[0].numpy() == 1), [1,1,1])
    test_res.at[0,'DICE_1'] = metrics.compute_dice_coefficient((label[0] == 1), (pred[0].numpy() == 1))
    test_res.at[0,'Hausdorff95_1'] = metrics.compute_robust_hausdorff(distances, 95)
    # class 2
    distances = metrics.compute_surface_distances((label[0] == 2), (pred[0].numpy() == 2), [1,1,1])
    test_res.at[0,'DICE_2'] = metrics.compute_dice_coefficient((label[0] == 2), (pred[0].numpy() == 2))
    test_res.at[0,'Hausdorff95_2'] = metrics.compute_robust_hausdorff(distances, 95)
    # class 3
    distances = metrics.compute_surface_distances((label[0] == 3), (pred[0].numpy() == 3), [1,1,1])
    test_res.at[0,'DICE_3'] = metrics.compute_dice_coefficient((label[0] == 3), (pred[0].numpy() == 3))
    test_res.at[0,'Hausdorff95_3'] = metrics.compute_robust_hausdorff(distances, 95)
    # class 4
    distances = metrics.compute_surface_distances((label[0] == 4), (pred[0].numpy() == 4), [1,1,1])
    test_res.at[0,'DICE_4'] = metrics.compute_dice_coefficient((label[0] == 4), (pred[0].numpy() == 4))
    test_res.at[0,'Hausdorff95_4'] = metrics.compute_robust_hausdorff(distances, 95)
    # class 5
    distances = metrics.compute_surface_distances((label[0] == 5), (pred[0].numpy() == 5), [1,1,1])
    test_res.at[0,'DICE_5'] = metrics.compute_dice_coefficient((label[0] == 5), (pred[0].numpy() == 5))
    test_res.at[0,'Hausdorff95_5'] = metrics.compute_robust_hausdorff(distances, 95)
    
    return test_res
  
  res_pivot = pd.DataFrame()
for i in range(0, len(testing_set)):
    sample = testing_set[i]
    patch_size = 64, 64, 64 
    patch_overlap = 20
    grid_sampler = torchio.inference.GridSampler(
        sample,
        patch_size,
        patch_overlap,
    )
    patch_loader = torch.utils.data.DataLoader(
        grid_sampler, batch_size=8)
    aggregator = torchio.inference.GridAggregator(grid_sampler)

    model.eval()
    with torch.no_grad():
        for patches_batch in patch_loader:
            inputs = patches_batch[MRI][DATA].to(device)
            locations = patches_batch['location']
            logits = model(inputs.float())
            labels = logits.argmax(dim=1, keepdim=True)
            aggregator.add_batch(labels, locations)
            
            
    pred = aggregator.get_output_tensor()
    label = prepare_aseg(testing_set[i][LABEL][DATA])
    plot_central_cuts(pred)
    temp_df = validate(label, pred)
    res_pivot = res_pivot.append(temp_df, ignore_index =True)

res_pivot

