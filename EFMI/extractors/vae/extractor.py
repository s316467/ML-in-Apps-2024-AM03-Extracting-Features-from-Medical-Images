import torch
import numpy as np


# extract latent vectors for the entire dataset
def extract_latent_vectors(model, dataloader, device):
    latent_vectors = []
    labels = []
    model.eval()
    with torch.no_grad():
        for data, target, _, _ in dataloader:
            data = data.to(device)
            target = target.to(device)
            mu, logvar = model.encode(data)
            latent_vector = model.reparameterize(mu, logvar)
            latent_vectors.append(latent_vector.cpu().numpy())
            labels.append(target.cpu().numpy())
    return np.concatenate(latent_vectors), np.concatenate(labels)


def extract_latents(model, dataloader, device, vae_type):
    latent_vectors, labels = extract_latent_vectors(model, dataloader, device)

    np.save(f"{vae_type}_latents.npy", latent_vectors)
    np.save(f"{vae_type}_labels.npy", labels)

    return (latent_vectors, labels)
