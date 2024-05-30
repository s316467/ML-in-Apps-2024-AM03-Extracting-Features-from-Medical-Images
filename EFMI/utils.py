import torch
import numpy as np
from tqdm.auto import tqdm
from torch import cuda

def extract_latent_vectors(model, dataloader, model_name):
    '''
        Extract latent vectors from the dataset and saves them as .npy.
        Takes as input model, dataloader, and model_name to appropriatly name
        the saved features and labels.
    '''

    device = 'cuda' if cuda.is_available() else 'cpu'
    steps = len(dataloader)
    progress_bar = tqdm(range(steps))

    latent_vectors = []
    labels = []
    model.eval()
    with torch.no_grad():
        count = 0
        for data,target,_,_ in dataloader:
            count += 1
            data = data.to(device)
            target = target.to(device)
            mu, logvar = model.encode(data)
            latent_vector = model.reparameterize(mu, logvar)
            latent_vectors.append(latent_vector.cpu().numpy())
            labels.append(target.cpu().numpy())

            if count == 100 or count==1 or count==10:#
                count = 0#
                np.save(f'features_{model_name}.npy', np.concatenate(latent_vectors))
                np.save(f'labels_{model_name}.npy', np.concatenate(labels))
                print('saving')

            progress_bar.update(1)
    return np.concatenate(latent_vectors), np.concatenate(labels)