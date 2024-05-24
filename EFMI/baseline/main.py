import argparse
import torch
from torch.utils.data import DataLoader
from dataset.PatchedDataset import PatchedDataset
from resnet import Resnet50Extractor
import svm


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PatchedDataset(root_dir=args.root_dir, num_images=args.num_images)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    resnet_extractor = Resnet50Extractor()

    features, labels = resnet_extractor.extract_features(dataloader)

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

    args = parser.parse_args()

    main(args)