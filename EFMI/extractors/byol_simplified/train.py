import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from train import train_byol
from dataset.PatchedDataset import PatchedDataset
from model.BYOL import BYOLNetwork
from model.augmentations import BYOLTransform


# 3. Training Loop
def byol_loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def train_byol(model, dataloader, epochs, device):
    print(f"Training BYOL for {epochs} epochs...")
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}"):
            optimizer.zero_grad()
            x1, x2 = batch
            x1, x2 = x1.to(device), x2.to(device)

            online_proj_1, online_proj_2, target_proj_1, target_proj_2 = model(x1, x2)

            loss = (
                byol_loss_fn(online_proj_1, target_proj_2)
                + byol_loss_fn(online_proj_2, target_proj_1)
            ).mean()

            loss.backward()
            optimizer.step()

            # Update target network
            tau = 0.996
            for online_params, target_params in zip(
                model.online_encoder.parameters(), model.target_encoder.parameters()
            ):
                target_params.data = (
                    tau * target_params.data + (1 - tau) * online_params.data
                )

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

        if (epoch + 1) % 2 == 0 and epoch > 0:
            torch.save(
                model.state_dict(),
                f"./results/byol-simplified/models/byol_{epoch}.pth",
            )

    return model


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PatchedDataset(
        root_dir=args.root_dir, num_images=args.num_images, transform=BYOLTransform()
    )

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=8
    )

    model = BYOLNetwork(projection_size=args.latent_dim).to(device)

    byol = train_byol(model, dataloader, epochs=args.num_epochs, device=device)

    torch.save(
        byol.state_dict(),
        f"./results/byol-simplified/models/byol_{args.num_epochs}.pth",
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
        "--num_epochs",
        type=int,
        default=10,
        help="Whether to finetune the pretrained dino and number of epochs",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="./results/byol-simplified/",
        help="Name of the experiment, save results in this path.",
    )
    args = parser.parse_args()

    main(args)
