import pickle

# Carica i latent vectors da file
with open('latent_vectors.pkl', 'rb') as f:
    latent_vectors = pickle.load(f)

print(latent_vectors)