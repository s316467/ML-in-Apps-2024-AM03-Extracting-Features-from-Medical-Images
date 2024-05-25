import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from baseline.extractor import BaselineExtractor


class Resnet50Extractor(BaselineExtractor):
    def __init__(self, latent_dim):
        super().__init__(
            model=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
            reduction_layer=nn.Linear(2048, latent_dim),
            model_name="Resnet50",
            latent_dim=latent_dim,
        )
