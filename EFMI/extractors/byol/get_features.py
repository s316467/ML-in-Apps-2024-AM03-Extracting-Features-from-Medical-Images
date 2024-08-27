import os
import argparse
import pickle
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from dataset.patches_dataset import PatchedDatasetAugmented
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from dataset.patches_dataset import train_test_split_dataset    
import config

def save_features(file_path, train_X, train_y, test_X, test_y):
    with open(file_path, 'wb') as f:
        pickle.dump((train_X, train_y, test_X, test_y), f, protocol=4)
    print(f"Features saved to {file_path}")

def load_features(file_path):
    with open(file_path, 'rb') as f:
        train_X, train_y, test_X, test_y = pickle.load(f)
    print(f"Features loaded from {file_path}")
    return train_X, train_y, test_X, test_y


import torch
import numpy as np

def apply_pca(train_X, test_X, n_components=128):
    pca = PCA(n_components=n_components)
    train_X_reduced = pca.fit_transform(train_X)
    test_X_reduced = pca.transform(test_X)
    print(f"PCA reduced features to shape: {train_X_reduced.shape}")
    return train_X_reduced, test_X_reduced

def inference(loader, model, device):
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            h = model(x)

        h = h.squeeze()
        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 5 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(model, train_loader, test_loader, device):
    train_X, train_y = inference(train_loader, model, device)
    test_X, test_y = inference(test_loader, model, device)
    return train_X, train_y, test_X, test_y


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, required=True, help='Path to your folder of images for self-supervised learning')
    parser.add_argument("--n_components", type=int, default=128, help="Number of components to keep after PCA")
    args = parser.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    dataset = PatchedDatasetAugmented(args.image_folder, size=config.IMAGE_SIZE, eval=True)

    # data loaders
    train_dataset, test_dataset = train_test_split_dataset(dataset, test_size=0.2)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        drop_last=True,
        num_workers=config.NUM_WORKERS,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        drop_last=True,
        num_workers=config.NUM_WORKERS,
    )

    # pre-trained model
 
    resnet = models.resnet50(pretrained=False)


    checkpoint_folder = Path(config.CHECKPOINT_FOLDER)
    checkpoints = list(checkpoint_folder.glob("checkpoint*.pt"))
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split(".")[1]))
        print(f"Loading checkpoint from {latest_checkpoint}")
    else:
        print("No checkpoints found")
        raise FileNotFoundError("No checkpoints found")
    
    
         

    resnet.load_state_dict(torch.load(latest_checkpoint, map_location=device))
    resnet = resnet.to(device)

    num_features = list(resnet.children())[-1].in_features


    # throw away fc layer
    resnet = nn.Sequential(*list(resnet.children())[:-1])
        # Add a new linear layer to reduce the feature dimension to 128
    

    n_classes = 2


    # fine-tune model

    feature_file = "extractors/byol/embeddings/eval_embeddings.pt" 
    # compute features (only needs to be done once, since it does not backprop during fine-tuning)
    if not os.path.exists(feature_file):
        print("### Creating features from pre-trained model ###")
        (train_X, train_y, test_X, test_y) = get_features(
            resnet, train_loader, test_loader, device
        )

        # apply PCA
        train_X, test_X = apply_pca(train_X, test_X, n_components=args.n_components)


        pickle.dump(
            (train_X, train_y, test_X, test_y), open(feature_file, "wb"), protocol=4
        )
        save_features(feature_file, train_X, train_y, test_X, test_y)
    else:
        print("### Loading features from file ###")
        train_X, train_y, test_X, test_y = load_features(feature_file)
    
    print("Shape of train_X: ", train_X.shape)
    print("Shape of test_X: ", test_X.shape)
    print("Shape of test_y: ", test_y.shape)
    print("Shape of train_y: ", train_y.shape)
    
    
