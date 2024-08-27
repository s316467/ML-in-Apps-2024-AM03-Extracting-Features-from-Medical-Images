import os
import argparse
from pathlib import Path
import torch
from torchvision import models
from torch.utils.data import DataLoader
import config  # Importiamo la configurazione
from dataset.patches_dataset import PatchedDatasetAugmented, train_test_split_dataset
from byol_pytorch.trainer import BYOLTrainer
from augmentations import get_train_augmentations


def main(args):
    # Carica il modello
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Carica il dataset
    ds = PatchedDatasetAugmented(args.image_folder, size=config.IMAGE_SIZE)
    train_ds, test_ds = train_test_split_dataset(ds, test_size=0.2)

    # Ottenere le trasformazioni basate sulla configurazione e l'indice specificato
    transform1, transform2 = get_train_augmentations(
        config.IMAGE_SIZE,
    )

    # Configurazione dei parametri BYOL
    byol_args = {
        "projection_size": config.PROJECTION_SIZE,
        "projection_hidden_size": config.PROJECTION_HIDDEN_SIZE,
        "moving_average_decay": 0.99,
        "augment_fn": transform1,
        "augment_fn2": transform2,
    }

    # Configura e avvia il trainer
    trainer = BYOLTrainer(
        net=resnet,
        dataset=train_ds,
        image_size=config.IMAGE_SIZE,
        hidden_layer="avgpool",
        learning_rate=config.LR,
        num_train_steps=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        checkpoint_every=config.CHECKPOINT_EVERY,
        checkpoint_folder=config.CHECKPOINT_FOLDER,
        resume_from_checkpoint=config.RESUME_FROM_CHECKPOINT,
        byol_kwargs=byol_args,
    )

    trainer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BYOL Lightning Training")
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Path to your folder of images for self-supervised learning",
    )

    args = parser.parse_args()

    main(args)
