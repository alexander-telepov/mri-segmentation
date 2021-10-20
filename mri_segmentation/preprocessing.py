import numpy as np
import torchio as tio
from torchio import transforms


def get_transforms(use_baseline=True):
    if use_baseline:
        train_transform = transforms.Compose([
            transforms.Crop((49, 22, 49, 47, 19, 28)),
            transforms.Pad(4)
        ])
        validation_transform = None

        return train_transform, validation_transform

    LI_LANDMARKS = np.array([
        0.0,
        8.06305571158,
        15.5085721044,
        18.7007018006,
        21.5032879029,
        26.1413278906,
        29.9862059045,
        33.8384058795,
        38.1891334787,
        40.7217966068,
        44.0109152758,
        58.3906435207,
        100.0
    ])
    LI_LANDMARKS = np.array([float(n) for n in LI_LANDMARKS.split()])
    landmarks_dict = {'MRI': LI_LANDMARKS}
    
    to_canonical = tio.ToCanonical()
    resample = tio.Resample((1., 1., 1.))
    get_foreground = tio.ZNormalization.mean
    standartizatize = tio.HistogramStandardization(landmarks_dict, masking_method=get_foreground)
    normalize = tio.ZNormalization(masking_method=get_foreground)

    training_transform = tio.Compose([
        to_canonical,
        resample,
        tio.RandomAnisotropy(p=0.25),
        tio.RandomGamma(p=0.2),
        standartizatize,
        normalize,
        tio.OneOf({
            tio.RandomBlur(p=0.5),
            tio.RandomNoise(p=0.5),
        }, p=0.5),
        tio.RandomElasticDeformation(p=0.05),
        tio.RandomAffine(p=0.8),
        tio.RandomFlip(p=0.5),
        tio.RandomBiasField(p=0.125),
        tio.OneOf({
            tio.RandomMotion(): 1,
            tio.RandomSpike(): 2,
            tio.RandomGhosting(): 2,
        }, p=0.125)
    ])

    testing_transform = tio.Compose([
        to_canonical,
        resample,
        standartizatize,
        normalize
    ])

    return training_transform, testing_transform
