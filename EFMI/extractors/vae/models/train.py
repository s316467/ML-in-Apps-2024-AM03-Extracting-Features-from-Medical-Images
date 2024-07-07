import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.plotting import plot_losses
from models.res_vae import ResVAE
from models.vae import VAE


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(dataloader, latent_dim, num_epochs, vae_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if vae_type == "vae":
        model = VAE(latent_dim).to(device)
        model = train_vae(model, dataloader, device, num_epochs, vae_type)
    if vae_type == "resvae":
        model = ResVAE(latent_dim).to(device)
        model = train_resvae(
            model, dataloader, device, latent_dim, num_epochs, vae_type
        )

    return model


def train_vae(model, dataloader, device, num_epochs, vae_type):

    current_epochs = 0

    try:
        model.load_state_dict(torch.load(f"{vae_type}_model_{current_epochs}.pth"))
        print("Model loaded correctly!")
    except:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, _, _, _) in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch+current_epochs+1}/{num_epochs}")
        ):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(
            f"Epoch {epoch+current_epochs+1}/{num_epochs}, Loss: {total_loss / len(dataloader.dataset):.4f}"
        )
        if (epoch + 1) % 2 == 0 and epoch > 0:
            torch.save(model.state_dict(), f"{vae_type}_{epoch+current_epochs+1}.pth")

    # Save the trained model
    torch.save(model.state_dict(), f"{vae_type}_100.pth")
    return model


# define a weighted funtion for beta
def weight_beta(num_epochs, beta):
    if beta == 1:
        x = np.linspace(-6, 6, num_epochs)
        weights = (1 / (1 + np.exp(-x))) * 0.05
    else:
        weights = np.ones(num_epochs)
    return weights


def train_resvae(model, dataloader, device, num_epochs, vae_type):
    # Hyperparameters
    learning_rate = 1e-5
    beta = 1e-05

    checkpoint_path = f"{vae_type}_MSE_beta={beta}_lr={learning_rate}_checkpoint.pth"
    # checkpoint_path = 'ResVAE_MSE_30_beta=1e-05_lr=1e-05.pth'

    # Initialize model and optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load checkpoint if available
    start_epoch = 0
    total_loss = 0

    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        total_loss = checkpoint.get("loss", 0)
        print("Model loaded correctly from checkpoint!")
        print(f"Starting from epoch {start_epoch}")
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch.")

    progress_bar = tqdm(range(len(dataloader) * (num_epochs - start_epoch)))
    loss_history = []
    mse_history = []
    kld_history = []

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        for data, _, _, _ in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            MSE, KLD = model.loss_function(recon_batch, data, mu, logvar)
            loss = MSE + beta * KLD
            loss_history.append(loss.item())
            kld_history.append(KLD.item())
            mse_history.append(MSE.item())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.update(1)

        # Save checkpoint every epoch
        if (epoch + 1) % 1 == 0:
            avg_loss = epoch_loss / len(dataloader.dataset)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                f"{vae_type}_MSE_beta={beta}_lr={learning_rate}_checkpoint.pth",
            )

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
            plot_losses(loss_history, mse_history, kld_history, beta)

    # Close the tqdm progress bar
    progress_bar.close()

    # Save the final model
    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        },
        f"{vae_type}_MSE_{num_epochs}_beta={beta}_lr={learning_rate}.pth",
    )
    return model
