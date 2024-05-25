import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

from baseline.extractor import BaselineExtractor


class DensenetExtractor(BaselineExtractor):
    def __init__(self, latent_dim):
        super().__init__(
            model=densenet121(weights=DenseNet121_Weights.DEFAULT),
            reduction_layer=nn.Linear(1024 * 7 * 7, latent_dim),
            model_name="Densenet121",
            latent_dim=latent_dim,
        )
