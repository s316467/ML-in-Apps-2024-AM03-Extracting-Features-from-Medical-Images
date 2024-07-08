import numpy as np

# Load the .npy files
train_embeddings = np.load('train_embeddings.npy')
train_labels = np.load('train_labels.npy')
val_embeddings = np.load('val_embeddings.npy')
val_labels = np.load('val_labels.npy')

# Print the shape of the arrays to understand their structure
print("Train Embeddings Shape:", train_embeddings.shape)
print("Train Labels Shape:", train_labels.shape)
print("Validation Embeddings Shape:", val_embeddings.shape)
print("Validation Labels Shape:", val_labels.shape)

# Print the first few entries to inspect
print("Train Embeddings:", train_embeddings[:5])
print("Train Labels:", train_labels[:5])
print("Validation Embeddings:", val_embeddings[:5])
print("Validation Labels:", val_labels[:5])