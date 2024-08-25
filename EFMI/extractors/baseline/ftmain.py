import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from train import fine_tune
from models.densenet import DensenetExtractor
from models.resnet import Resnet50Extractor
from dataset.PatchedDataset import PatchedDataset
import classifier.svm as svm
import torchvision.models as models
from torchvision.models import ResNet50_Weights

def create_binary_classifier():
    # Load a pretrained ResNet50 model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    # Freeze all the parameters in the network
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the last fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 2)  # 2 output classes for binary classification
    )
    
    return model


def train_test_split_loaders(full_dataset, train_ratio):
    
    train_size = int(train_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


def main(args):
    dataset = PatchedDataset(root_dir=args.root_dir, num_images=args.num_images)
    
    # Create the model
    extractor = create_binary_classifier()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    extractor = extractor.to(device)

    print(f"Loaded {args.model_name}")

    train_loader, test_loader = train_test_split_loaders(dataset, 0.8)

    print(f"Finetuning {args.model_name}..")
    extractor = fine_tune(extractor, train_loader, args.ft_epochs, args.model_name)

    print(f"Extracting feature from finetuned {args.model_name}..")
    train_features, train_labels = extractor.extract_features(train_loader)
    test_features, test_labels = extractor.extract_features(test_loader)
    
    svm.classify_with_provided_splits(
        train_features,
        train_labels,
        test_features,
        test_labels,
        args.results_path,
        with_pca=True,
        pca_components=args.latent_dim,
    )


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
        "--ft_epochs",
        type=int,
        default=10,
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