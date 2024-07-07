import numpy as np
import torch
from tqdm import tqdm


# Function to extract features using the fine-tuned model
def extract_features(dataloader, model):
    print("Extracting features...")
    steps = len(dataloader)
    progress_bar = tqdm(total = steps)

    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, label, p_id, coords in dataloader:
            inputs = inputs.cuda()
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
            labels.append(label.numpy())
            progress_bar.update(1)
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels
