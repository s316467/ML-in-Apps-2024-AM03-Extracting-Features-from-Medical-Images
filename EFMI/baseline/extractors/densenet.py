from torchvision.models import densenet121, DenseNet121_Weights

from base_extractor import BaselineExtractor

class DensenetExtractor(BaselineExtractor):
    def __init__(self, latent_dim):
        super().__init__(
            model=densenet121(weights=DenseNet121_Weights.DEFAULT),
            output_dim=latent_dim,
            reduction_dim=1024,
            model_name="Densenet121",
        )
