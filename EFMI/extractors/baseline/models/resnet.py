import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from extractors.baseline.extractor import BaselineExtractor


class Resnet50Extractor(BaselineExtractor):
    def __init__(
        self,
        model=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
        latent_dim=128,
        verbose=False,
    ):
        super().__init__(
            model=model,
            reduction_layer=nn.Linear(2048, latent_dim),
            model_name="Resnet50",
            latent_dim=latent_dim,
            verbose=verbose,
        )


def get_adapted_resnet50():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 2)
    )

    return model
