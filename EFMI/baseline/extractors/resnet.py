from torchvision.models import resnet50, ResNet50_Weights

from base_extractor import BaselineExtractor


class Resnet50Extractor(BaselineExtractor):
    def __init__(self):
        super().__init__(
            model=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
            output_dim=512,
            reduction_dim=2048,
            model_name="Resnet50",
        )
