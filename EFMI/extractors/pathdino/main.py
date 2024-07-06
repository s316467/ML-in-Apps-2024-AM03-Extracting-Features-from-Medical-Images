import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from pathdino.extractor import extract_features
import classifier.svm as svm
from EFMI.utils.plotting import *
from dataset.PatchedDataset import PatchedDataset
from model.PathDino import get_pathDino_model
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import random_split

custom_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(
            lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x
        ),  # ensure 3 channels
        transforms.Normalize(
            mean=[0.7598, 0.6070, 0.7159], std=[0.1377, 0.1774, 0.1328]
        ),
    ]
)


def fine_tune(model, dataset, num_epochs, batch_size):
    criterion = nn.CrossEntropyLoss()  # ????
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()  # Mixed precision training scaler

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = model.cuda()

    for param in model.parameters():
        param.requires_grad = True

    model.train()
    training_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(
                non_blocking=True
            )
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            if torch.isnan(loss):
                print(f"NaN loss encountered at epoch {epoch+1}")
                continue
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        training_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")

    # Plot and save the training loss
    plot_training_loss(training_losses, ".")

    return model


def main(args):

    pathdino, dino_transform = get_pathDino_model(
        weights_path=args.pretrained_dino_path
    )

    # choose transform (Custom vs Dino) ?
    dataset = PatchedDataset(
        root_dir=args.root_dir, num_images=args.num_images, transform=custom_transform
    )

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    if args.fine_tune:
        pathdino = fine_tune(pathdino, dataset, args.batch_size, args.fine_tune_epochs)

    features, labels = extract_features(dataloader)

    svm.classify(
        features, labels, args.experiment_name, with_pca=True, pca_components=128
    )

    plot_tsne(
        test_vectors_pca,
        y_pred,
        title="t-SNE plot of SVM predictions",
        filename="tsne_plot.png",
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
        "--pretrained_dino_path", type="str", help="PathDino pretrained weights path"
    )
    parser.add_argument(
        "--fine_tune",
        action="store_true",
        help="Wheter to finetune the pretrained dino",
    )
    parser.add_argument(
        "--fine_tune_epochs",
        type=int,
        default=10,
        help="Number of epochs for finetuning",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    main(args)
