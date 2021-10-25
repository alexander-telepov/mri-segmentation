import torch
import numpy as np
from surface_distance import metrics as _metrics


def dice_score(prediction, targets, n_classes, channel=0):
    prediction = torch.softmax(prediction, dim=1)

    assert prediction.size() == targets.size(), \
        'predict {} & target {} shape do not match'.format(prediction.size(), targets.size())

    dice = _metrics.compute_dice_coefficient((np.argmax(targets[0].cpu().numpy(), axis=0) == channel),
                                             (np.argmax(prediction[0].cpu().numpy(), axis=0) == channel))

    return dice


def hausdorff_score(prediction, targets, spacing, channel=0, level=95):
    distances = _metrics.compute_surface_distances((np.argmax(targets.squeeze(0).cpu().numpy(), axis=0) == channel),
                                                   (np.argmax(prediction.squeeze(0).cpu().numpy(), axis=0) == channel),
                                                   spacing)
    return _metrics.compute_robust_hausdorff(distances, level)
