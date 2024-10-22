import torch
import torch.nn as nn
import numpy as np


class BaselineExtractor:
    def __init__(self, model, reduction_layer, model_name, latent_dim, verbose=False):

        self.model_name = model_name

        self.model = nn.Sequential(*list(model.children())[:-1])
        self.reduction_layer = reduction_layer

        for param in self.model.parameters():
            param.requires_grad = False

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.reduction_layer = self.reduction_layer.to(self.device)
        self.latent_dim = latent_dim
        self.verbose = verbose

    def extract_features(self, data_loader):
        if self.verbose:
            print(f"Extracting features from {self.model_name}...")

        self.model.eval()
        self.reduction_layer.eval()
        features = []
        labels = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                image, label, patient_id, coordinates = batch
                image = image.to(self.device)

                if self.verbose:
                    print(f"Extracting features for batch {batch_idx+1}")

                features_batch = self.model(image).squeeze()
                features_batch = features_batch.view(features_batch.size(0), -1)
                features_batch = self.reduction_layer(features_batch)

                features.append(features_batch.cpu().numpy())
                labels.append(label.numpy())

        features = np.concatenate(features)
        labels = np.concatenate(labels)

        print("Saving features as numpy arrays...")
        np.save(f"./{self.model_name}_features_{self.latent_dim}.npy", features)
        np.save(f"./{self.model_name}_labels_{self.latent_dim}.npy", labels)

        return features, labels
