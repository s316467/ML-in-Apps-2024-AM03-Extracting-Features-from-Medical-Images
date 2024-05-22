import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

# Load the latent vectors from the .pkl file
with open('latent_vectors.pkl', 'rb') as file:
    data = pickle.load(file)

# Ensure the tensors are on CPU and convert them to NumPy arrays
def to_numpy(tensor):
    return tensor.cpu().numpy()

# Assuming the keys in the .pkl file are 'w0', 'w1', 'wt0', and 'wt1'
w0 = to_numpy(data['w0'])
w1 = to_numpy(data['w1'])
wt0 = to_numpy(data['wt0'])
wt1 = to_numpy(data['wt1'])

# Flatten the tensors to 2D
w0_flat = w0.reshape((w0.shape[0], -1))
w1_flat = w1.reshape((w1.shape[0], -1))
wt0_flat = wt0.reshape((wt0.shape[0], -1))
wt1_flat = wt1.reshape((wt1.shape[0], -1))

# Combine the tensors into a single dataset
# Optionally, you can label them for visualization purposes
labels = ['w0'] * w0_flat.shape[0] + ['w1'] * w1_flat.shape[0] + ['wt0'] * wt0_flat.shape[0] + ['wt1'] * wt1_flat.shape[0]
latent_vectors = np.concatenate([w0_flat, w1_flat, wt0_flat, wt1_flat])

# Apply t-SNE to reduce dimensionality with a suitable perplexity value
perplexity_value = min(30, len(latent_vectors) - 1)  # Ensure perplexity is less than number of samples
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)
latent_vectors_2d = tsne.fit_transform(latent_vectors)

# Visualize the result using matplotlib
plt.figure(figsize=(10, 7))
unique_labels = set(labels)
colors = ['red', 'blue', 'green', 'purple']
for label, color in zip(unique_labels, colors):
    indices = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(latent_vectors_2d[indices, 0], latent_vectors_2d[indices, 1], label=label, color=color)

plt.legend()
plt.title('t-SNE Visualization of Latent Vectors')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()
