from unet import UNet
from .models.hrnet import hrnet18
from .models.attention import AttentionUnet


def get_model(architecture='unet', device='cpu', **kwargs):
    if architecture == 'hrnet18':
        model = hrnet18
    elif architecture == 'attention_unet':
        model = AttentionUnet
    elif architecture == 'unet':
        model = UNet
    else:
        raise ValueError(f'Unknown architecture: {architecture}')

    model = model(**kwargs).to(device)

    return model
