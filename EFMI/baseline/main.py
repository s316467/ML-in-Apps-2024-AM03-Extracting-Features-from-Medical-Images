import argparse
from torch.utils.data import DataLoader

from extractors.densenet import DensenetExtractor
from extractors.resnet import Resnet50Extractor

from dataset.PatchedDataset import PatchedDataset
import svm


def load_extractor(model_name="resnet50", latent_dim=512):
    supported_models = ["densenet121", "resnet50"]

    if model_name not in supported_models:
        raise ValueError(
            f"Unsupported model name '{model_name}'. Supported models: {supported_models}"
        )

    if model_name == "densenet121":
        return DensenetExtractor(latent_dim)
    elif model_name == "resnet50":
        return Resnet50Extractor(latent_dim)


def main(args):
    dataset = PatchedDataset(root_dir=args.root_dir, num_images=args.num_images)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    extractor = load_extractor(args.baseline_model, args.output_dim)

    features, labels = extractor.extract_features(dataloader)

    svm.classify(features, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract patches features with a pre-trained resnet50"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Path to directory containing the patches folders",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        help="Which pretrained baseline model to use as baseline feature extractor, defaults to resnet50. Availables: resnet50, densenet121",
    )
    parser.add_argument(
        "--output_dim",
        type=int,
        default=512,
        help="Extracted latent vector dimension, defaults to 512",
    )
    args = parser.parse_args()

    main(args)
