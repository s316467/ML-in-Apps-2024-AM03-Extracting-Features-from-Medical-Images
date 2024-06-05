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
from torchvision import transforms

# Define a transformation to ensure 3 channels (RGB)
transformInput = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Handle grayscale images
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to load images and extract features
def load_images_and_extract_features(folder_path, model, transform):
    images = []
    labels = []
    
    for label, subfolder in enumerate(['in_roi_patches', 'not_roi_patches']):
        subfolder_path = os.path.join(folder_path, subfolder)
        
        for subsubfolder in os.listdir(subfolder_path):
            subsubfolder_path = os.path.join(subfolder_path, subsubfolder)
            
            if os.path.isdir(subsubfolder_path):
                for filename in os.listdir(subsubfolder_path):
                    if filename.endswith('.png'):
                        img_path = os.path.join(subsubfolder_path, filename)
                        img = Image.open(img_path).convert("RGB")
                        img_tensor = transform(img)
                        embedding = model(img_tensor.unsqueeze(0)).detach().numpy().flatten()
                        
                        images.append(embedding)
                        labels.append(label)
                
    return np.array(images), np.array(labels)

# Function to plot t-SNE and save to a file
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

def plot_pca_variance(pca, output_dir):
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance Explained')
    plt.title('PCA - Variance Explained by Components')
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'pca_variance.png'))
    plt.close()

# Load the model and transformation function
model, _ = get_pathDino_model(weights_path='./inference/PathDino512.pth')

# Load images and extract features
folder_path = './CRC_WSIs_no_train_test'
latent_vectors, labels = load_images_and_extract_features(folder_path, model, transformInput)

# Apply PCA to reduce dimensionality to 128
pca = PCA(n_components=128)
latent_vectors_pca = pca.fit_transform(latent_vectors)

# Plot PCA variance
plot_pca_variance(pca, '.')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(latent_vectors_pca, labels, test_size=0.2, random_state=42)

# Initialize and train the SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot t-SNE of the PCA-reduced features and save to file
plot_tsne(X_test, y_pred, title="t-SNE plot of SVM predictions", filename="tsne_plot.png")

"""
# Testing with a new image
test_image_path = './inference/img.png'
test_image = Image.open(test_image_path).convert("RGB")
test_image_tensor = transformInput(test_image)
test_embedding = model(test_image_tensor.unsqueeze(0)).detach().numpy().flatten()
test_embedding_pca = pca.transform([test_embedding])
"""
# print(test_embedding_pca.shape)

"""
All patients all embeddings:
Accuracy: 0.8864751226348984
Classification Report:
               precision    recall  f1-score   support

           0       0.88      0.89      0.88      1376
           1       0.90      0.88      0.89      1478

    accuracy                           0.89      2854
   macro avg       0.89      0.89      0.89      2854
weighted avg       0.89      0.89      0.89      2854
"""
