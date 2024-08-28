import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from extractors.baseline.extractor import BaselineExtractor


class Resnet50Extractor(BaselineExtractor):
    def __init__(self, latent_dim, verbose=False):
        super().__init__(
            model=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
            reduction_layer=nn.Linear(2048, latent_dim),
            model_name="Resnet50",
            latent_dim=latent_dim,
            verbose=verbose,
        )


def get_adapted_resnet50(latent_dim=128):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    model = nn.Sequential(*list(model.children())[:-1])

    model = nn.Sequential(
        *list(model.children()), nn.Flatten(), nn.Linear(2048, latent_dim)
    )

    for param in model.parameters():
        param.requires_grad = False

    for param in model[-1].parameters():
        param.requires_grad = True

    return model
