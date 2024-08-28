import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class AdaptedResnet50Extractor:
    def __init__(self, model, latent_dim, verbose=False):
        self.model = model
        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.verbose = verbose

    def extract_features(self, data_loader):
        self.model.eval()
        features = []
        labels = []

        with torch.no_grad():
            for batch in tqdm(
                data_loader, desc="Extracting features", disable=not self.verbose
            ):
                images, batch_labels = batch[0], batch[1]
                images = images.to(self.device)

                feature_vectors = self.model(images)

                features.append(feature_vectors.cpu().numpy())
                labels.append(batch_labels.numpy())

        features = np.concatenate(features)
        labels = np.concatenate(labels)

        if self.verbose:
            print(f"Extracted features shape: {features.shape}")
            print(f"Labels shape: {labels.shape}")

        return features, labels
