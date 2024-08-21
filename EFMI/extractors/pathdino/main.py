import argparse
from train import fine_tune
from torch.utils.data import DataLoader
from extractor import extract_features
import classifier.svm as svm
from utils.plotting import *
from dataset.PatchedDataset import PatchedDataset
from model.PathDino import get_pathDino_model


def main(args):

    pathdino, dino_transform = get_pathDino_model(
        weights_path=args.pretrained_dino_path
    )

    pathdino.cuda()

    dataset = PatchedDataset(
        root_dir=args.root_dir, num_images=args.num_images, transform=dino_transform
    )

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=8
    )

    if args.fine_tune:
        print(f"Finetuning pathdino512 from {args.pretrained_dino_path}")
        pathdino = fine_tune(pathdino, dataloader, args.fine_tune_epochs)

    features, labels = extract_features(dataloader, pathdino)

    # TODO: choose different dim.red layer?
    # TODO: Pass train and test instead
    svm.classify(
        features,
        labels,
        args.results_path,
        with_pca=True,
        pca_components=args.latent_dim,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PathDino")
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
        "--latent_dim",
        type=int,
        default=128,
        help="Extracted latent vector dimension, defaults to 128",
    )
    parser.add_argument(
        "--pretrained_dino_path", 
        type=str,
        default="./extractors/pathdino/model/PathDino512.py" ,
        help="PathDino pretrained weights path"
    )
    parser.add_argument(
        "--fine_tune",
        type=str,
        default="finetune",
        help="Wheter to finetune the pretrained dino. finetune := yes",
    )
    parser.add_argument(
        "--fine_tune_epochs",
        type=int,
        default=10,
        help="Number of epochs for finetuning",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="./results/pathdino/",
        help="Name of the experiment, save results in this path.",
    )
    args = parser.parse_args()
    args.fine_tune = True if args.fine_tune == "finetune" else False
    main(args)
