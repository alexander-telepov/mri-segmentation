import matplotlib.pyplot as plt
import torch
import numpy as np
import nibabel
import nibabel as nib


def plot_central_cuts(img, title=""):
    """
    param image: tensor or np array of shape (CxDxHxW) if t is None
    """
    if isinstance(img, torch.Tensor):
        img = img.numpy()
        if len(img.shape) > 3:
            img = img[0, :, :, :]

    elif isinstance(img, nibabel.nifti1.Nifti1Image):
        img = img.get_fdata()

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(3 * 6, 6))
    axes[0].imshow(img[img.shape[0] // 2, :, :], cmap='gray')
    axes[1].imshow(img[:, img.shape[1] // 2, :], cmap='gray')
    axes[2].imshow(img[:, :, img.shape[2] // 2], cmap='gray')

    plt.show()


def plot_predicted(img, seg, delta=0, title=""):
    """
    param image: tensor or np array of shape (CxDxHxW) if t is None
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
        if len(img.shape) == 5:
            img = img[0, 0, :, :, :]
        elif len(img.shape) == 4:
            img = img[0, :, :, :]

    elif isinstance(img, nibabel.nifti1.Nifti1Image):
        img = img.get_fdata()

    if isinstance(seg, torch.Tensor):
        seg = seg[0].cpu().numpy().astype(np.uint8)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(3 * 4, 4))
    axes[0].imshow(img[img.shape[0] // 2 + delta, :, :])
    axes[1].imshow(seg[seg.shape[0] // 2 + delta, :, :])
    intersect = img[img.shape[0] // 2 + delta, :, :] + seg[seg.shape[0] // 2 + delta, :, :] * 100
    axes[2].imshow(intersect, cmap='gray')

    plt.show()


def get_bounds(self):
    """Get image bounds in mm.

    Returns:
        np.ndarray: [description]
    """
    first_index = 3 * (-0.5,)
    last_index = np.array(self.spatial_shape) - 0.5
    first_point = nib.affines.apply_affine(self.affine, first_index)
    last_point = nib.affines.apply_affine(self.affine, last_index)
    array = np.array((first_point, last_point))
    bounds_x, bounds_y, bounds_z = array.T.tolist()
    return bounds_x, bounds_y, bounds_z
