import torch
import numpy as np
from surface_distance import metrics as _metrics
from .loss import DiceLoss


def dice_score(prediction, targets, n_classes, channel=0):
    dice_loss = DiceLoss(n_classes)
    prediction = torch.softmax(prediction, dim=1)
    targets = dice_loss.one_hot_encoder(targets)

    assert prediction.size() == targets.size(), \
        'predict {} & target {} shape do not match'.format(prediction.size(), targets.size())

    dice = dice_loss.dice_loss(prediction[:, channel], targets[:, channel])

    return 1.0 - dice.item()


def hausdorff_score(prediction, targets, spacing, channel=0, level=95):
    distances = _metrics.compute_surface_distances((targets.squeeze(0).cpu().numpy() == channel),
                                                   (np.argmax(prediction.squeeze(0).cpu().numpy(), axis=0) == channel),
                                                   spacing)
    return _metrics.compute_robust_hausdorff(distances, level)
