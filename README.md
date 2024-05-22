# ML-in-Apps-2024-AM03-Extracting-Features-from-Medical-Images
Repository of Politecnico di Torino Master's degree course Machine Learning in Applications 


# dataset_patches.py
Example usage:

```python
from dataset_patches import patchesDataset

rootdir = 'patches' # path to your patches folder
dataset = patchesDataset(rootdir)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
```

The patchesDataset class applies the ToTensor transform, but supports passing as input (after the rootdir) a transform.Compose. If you want to add other transformations add again the ToTensor transform. Example:
```python
from dataset_patches import patchesDataset

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ### your other transforms ###
])
rootdir = 'patches' # path to your patches folder
dataset = patchesDataset(rootdir, transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
```
