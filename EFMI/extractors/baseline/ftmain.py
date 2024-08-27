import argparse
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from finetune import fine_tune
from models.resnet import get_adapted_resnet50, Resnet50Extractor
from dataset.PatchedDataset import PatchedDataset, train_test_split
import classifier.svm as svm


def main(args):
    dataset = PatchedDataset(root_dir=args.root_dir, num_images=args.num_images)

    resnet50 = get_adapted_resnet50().cuda()

    train_dataset, test_dataset = train_test_split(dataset, 0.8)

    np.save(f'finetune_resnet_{args.ft_epochs}_train_dataset.npy', train_array)
    np.save(f'finetune_resnet_{args.ft_epochs}_test_dataset.npy', test_array)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8
    )

    print(f"Finetuning {args.model_name}..")
    resnet50 = fine_tune(resnet50, train_loader, args.ft_epochs, args.model_name)

    extractor = Resnet50Extractor(model=resnet50)

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
        default=32,
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
