from unet import UNet
from .hrnet import hrnet18
from .attention import AttentionUnet


def get_model(architecture, device, **kwargs):
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
