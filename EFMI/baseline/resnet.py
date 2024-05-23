import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class Resnet50Extractor:
    def __init__(self):
        self.resnet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])
        for param in self.resnet50.parameters():
            param.requires_grad = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.resnet50 = self.resnet50.to(self.device)
        
    def extract_features(self, data_loader):
        self.resnet50.eval()
        features = []
        labels = []
        with torch.no_grad():
            for image, label, patient_id, _  in data_loader:
                image = image.to(self.device)
                features_batch = self.resnet50(image).squeeze()
                features.append(features_batch.cpu().numpy())
                labels.append(label.numpy())
                print("Feature extracted for image: ", patient_id)
        features = np.concatenate(features)
        labels = np.concatenate(labels)
        print(features.shape)
        print(labels.shape)
        
        return features, labels
