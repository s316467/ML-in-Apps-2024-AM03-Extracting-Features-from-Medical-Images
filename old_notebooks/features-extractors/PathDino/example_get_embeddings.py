import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PathDino import get_pathDino_model
from PIL import Image
import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

# Enable CUDA benchmarking
torch.backends.cudnn.benchmark = True

# Define a transformation to ensure 3 channels (RGB)
def ensure_three_channels(x):
    return x.repeat(3, 1, 1) if x.shape[0] == 1 else x

transformInput = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(ensure_three_channels),  # Replace lambda with named function
    transforms.Normalize(mean=[0.7598, 0.6070, 0.7159], std=[0.1377, 0.1774, 0.1328])
])

# Custom dataset to load images and labels
class ImageDataset(Dataset):
    def __init__(self, folder_path, transform):
        self.folder_path = folder_path
        self.transform = transform
        self.images = []
        self.labels = []
        self.load_images()

    def load_images(self):
        for label, subfolder in enumerate(['in_roi_patches', 'not_roi_patches']):
            subfolder_path = os.path.join(self.folder_path, subfolder)
            for subsubfolder in os.listdir(subfolder_path):
                subsubfolder_path = os.path.join(subfolder_path, subsubfolder)
                if os.path.isdir(subsubfolder_path):
                    for filename in os.listdir(subsubfolder_path):
                        if filename.endswith('.png'):
                            img_path = os.path.join(subsubfolder_path, filename)
                            self.images.append(img_path)
                            self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Main function to guard multiprocessing code
if __name__ == "__main__":
    # Load the model and fine-tune it
    model, _ = get_pathDino_model(weights_path='./inference/PathDino512.pth')
    model = model.cuda()  # Move model to GPU if available

    # Ensure model parameters require gradients
    for param in model.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load the data
    train_dataset = ImageDataset(os.path.join('./CRC_WSIs_no_train_test', 'train'), transformInput)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=4)
    test_dataset = ImageDataset(os.path.join('./CRC_WSIs_no_train_test', 'test'), transformInput)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)

    # Function to plot loss during training
    def plot_training_loss(losses, output_dir):
        plt.figure()
        plt.plot(losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_dir, 'training_loss.png'))
        plt.close()

    # Mixed precision training scaler
    scaler = GradScaler()

    # Fine-tune the model
    num_epochs = 50
    model.train()
    training_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        training_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")

    # Plot and save the training loss
    plot_training_loss(training_losses, '.')

    # Function to extract features using the fine-tuned model
    def extract_features(dataloader, model):
        model.eval()
        features = []
        labels = []
        with torch.no_grad():
            for inputs, label in dataloader:
                inputs = inputs.cuda()
                outputs = model(inputs)
                features.append(outputs.cpu().numpy())
                labels.append(label.numpy())
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        return features, labels

    # Extract features for train and test data
    train_vectors, train_labels = extract_features(train_loader, model)
    test_vectors, test_labels = extract_features(test_loader, model)

    # Apply PCA to reduce dimensionality to 128
    pca = PCA(n_components=128)
    train_vectors_pca = pca.fit_transform(train_vectors)
    test_vectors_pca = pca.transform(test_vectors)

    # Plot PCA variance
    def plot_pca_variance(pca, output_dir):
        plt.figure()
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Variance Explained')
        plt.title('PCA - Variance Explained by Components')
        plt.grid()
        plt.savefig(os.path.join(output_dir, 'pca_variance.png'))
        plt.close()

    plot_pca_variance(pca, '.')

    # Initialize and train the SVM classifier
    svm_classifier = SVC(kernel='linear', C=1.0)
    svm_classifier.fit(train_vectors_pca, train_labels)

    # Predict on the test set
    y_pred = svm_classifier.predict(test_vectors_pca)

    # Evaluate the classifier
    accuracy = accuracy_score(test_labels, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(test_labels, y_pred))

    # Plot t-SNE of the PCA-reduced features and save to file
    def plot_tsne(features, labels, title, filename):
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(features)
        plt.figure(figsize=(10, 7))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', s=5)
        plt.colorbar()
        plt.title(title)
        plt.xlabel("t-SNE feature 1")
        plt.ylabel("t-SNE feature 2")
        plt.savefig(filename)
        plt.close()

    plot_tsne(test_vectors_pca, y_pred, title="t-SNE plot of SVM predictions", filename="tsne_plot.png")
