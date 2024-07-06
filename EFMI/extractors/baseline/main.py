import argparse
from torch.utils.data import DataLoader

from models.densenet import DensenetExtractor
from models.resnet import Resnet50Extractor
from dataset.PatchedDataset import PatchedDataset
import svm


def load_extractor(model_name="resnet50", latent_dim=512):
    supported_models = ["densenet121", "resnet50"]

    assert (
        model_name not in supported_models
    ), f"Unsupported model name '{model_name}'. Supported models: {supported_models}"

    if model_name == "densenet121":
        return DensenetExtractor(latent_dim)
    elif model_name == "resnet50":
        return Resnet50Extractor(latent_dim)


def main(args):
    dataset = PatchedDataset(root_dir=args.root_dir, num_images=args.num_images)

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    extractor = load_extractor(args.model_name, args.latent_dim)

    features, labels = extractor.extract_features(dataloader)

    svm.classify(features, labels, args.results_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract patches features with pre-trained nets"
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
        help="How may images to use (test purpose)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet50",
        help="Which pretrained baseline model to use as baseline feature extractor, defaults to resnet50. Availables: resnet50, densenet121",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=128,
        help="Extracted latent vector dimension, defaults to 128",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        help="Name of the experiment, save results in this path.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    main(args)
