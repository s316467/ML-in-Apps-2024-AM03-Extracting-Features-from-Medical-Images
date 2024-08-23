import argparse
from torch.utils.data import DataLoader, random_split

from models.ROIRegressor import ROIRegressor
from models.densenet import DensenetExtractor
from models.resnet import Resnet50Extractor
from dataset.ROIPatchedDataset import ROIPatchDataset
import torch


def load_extractor(model_name="resnet50", latent_dim=512):
    supported_models = ["densenet121", "resnet50"]

    assert (
        model_name in supported_models
    ), f"Unsupported model name '{model_name}'. Supported models: {supported_models}"

    if model_name == "densenet121":
        return DensenetExtractor(latent_dim)
    elif model_name == "resnet50":
        return Resnet50Extractor(latent_dim)


def train_test_split_loaders(full_dataset, train_ratio):
    train_size = int(train_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8
    )

    return train_loader, test_loader


def main(args):

    dataset = ROIPatchDataset(args.patches_path, num_images=args.num_images)

    train_loader, test_loader = train_test_split_loaders(dataset, 0.8)

    extractor = load_extractor(args.model_name, args.latent_dim)

    train_features, train_coords = extractor.extract_features(train_loader)
    test_features, test_coords = extractor.extract_features(test_loader)

    input_dim = train_features.shape[1]
    output_dim = train_coords.shape[1] * 2  # x and y for each coordinate

    model, test_loss = ROIRegressor.train_and_evaluate_regressor(
        train_features, train_coords, test_features, test_coords, input_dim, output_dim
    )

    # Save results
    torch.save(model.state_dict(), f"{args.results_path}/roi_regressor.pth")
    with open(f"{args.results_path}/roi_regression_results.txt", "w") as f:
        f.write(f"Test Loss: {test_loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract patches from WSI images.")
    parser.add_argument(
        "--patches_path", type=str, required=True, help="Path to the dataset directory"
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

    args = parser.parse_args()

    main(args)
