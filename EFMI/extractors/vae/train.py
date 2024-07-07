import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from res_vae import ResVAE
from model import VAE


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(dataloader, device, latent_dim, vae_type):

    current_epochs = 0

    if vae_type == "vae":
        model = VAE(latent_dim)
    if vae_type == "resvae":
        model = ResVAE(latent_dim)

    model.to(device)

    try:
        model.load_state_dict(torch.load(f"{vae_type}_model_{current_epochs}.pth"))
        print("Model loaded correctly!")
    except:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 1
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
