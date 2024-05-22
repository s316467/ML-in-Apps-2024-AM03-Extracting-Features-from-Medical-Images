import sys
import os

# Add the parent directory of 'torch_utils' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import torch

with open('..\\training_runs\\WIP_stylegan3_training-runs_00000-stylegan3-r-wsis-gpus1-batch16-gamma6.6_network-snapshot-000001.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
# Generate a latent vector
z_dim = G.z_dim
latent_vector = torch.randn([1, z_dim]).cuda()  # Generate a random latent vector

# Optionally, you can save this latent vector for future use
torch.save(latent_vector, 'latent_vector.pt')

# Generate an image using the latent vector
c = torch.zeros([1, G.c_dim]).cuda() if G.c_dim > 0 else None
img = G(latent_vector, c)  # NCHW, float32, dynamic range [-1, +1], no truncation

# Save or display the generated image
# Example: convert the tensor to a PIL image and save it
import torchvision.transforms as transforms
from PIL import Image

img = (img.permute(0, 2, 3, 1) * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
img = Image.fromarray(img[0].cpu().numpy(), 'RGB')
img.save('generated_image.png')

# Load and use a saved latent vector
# latent_vector = torch.load('latent_vector.pt')
# img = G(latent_vector.cuda(), c)