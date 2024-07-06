import numpy as np
import torch


# Function to extract features using the fine-tuned model
def extract_features(dataloader, model):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, label, p_id, coords in dataloader:
            inputs = inputs.cuda()
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
            labels.append(label.numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels
