# ML-in-Apps-2024-AM03-Extracting-Features-from-Medical-Images
Repository of Politecnico di Torino Master's degree course Machine Learning in Applications 

## Prerequisites
Install OpenSlide tools:
```sh
sudo apt-get install openslide-tools
```

## Extract patches from WSI images
To extract annotated patches from wsi images, run ```EFMI/launch_scripts/extract-patches.sh DATASET_PATH NOT_ROI_PATH ROI_PATH PATCH_SIZE MAG_LEVEL```. This requires that ```DATASET_PATH``` contains both svs image files and xml annotation files. The mapping svs -> xml is given by their names, i.e ```1.svs -> 1.xml```. This will extract patches and save them with this pattern: ```{x}_{y}_mag{mag_level}.png```


## Example usage of the PatchedDataset class
Note that patches must have this file name: ```{x}_{y}_mag{mag_level}.png```

```python
from dataset import PatchedDataset

rootdir = 'patches' # path to your patches folder
ds = PatchedDataset(rootdir)
dataloader = DataLoader(ds, batch_size=8, shuffle=True)
```

The PatchedDataset class applies the ToTensor transform, but supports passing as input (after the rootdir) a transform.Compose. If you want to add other transformations add again the ToTensor transform. Example:
```python
from dataset import PatchedDataset

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ### your other transforms ###
])
rootdir = 'patches' # path to your patches folder
ds = PatchedDataset(rootdir, transform)
dataloader = DataLoader(ds, batch_size=8, shuffle=True)
```


## Test the VAE latents with SVM classifier
To test the VAE latents with the svm classifier, simply run ```EFMI/launch_scripts/test-vae.sh root_dir num_images batch_size latent_dim```. ```root_dir``` is the dataset root directory containing the patches inside two distinct folders, in ROI and not in ROI. ```num_images``` (defaults to 24) specifies on how many images-patches to train the VAE, use this for test purposes (note that patches dirs contains image_name.svs/patch.png list). ```batch_size``` is the batch_size with which to train the VAE (defaults to 8), ```latent_dim``` specifies the extract latents dimension (defaults to 100).