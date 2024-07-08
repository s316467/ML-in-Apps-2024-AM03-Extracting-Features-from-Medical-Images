# Extracting features from medical images
Authors: Maria Scoleri, Antonio Ferrigno, Fabio Barbieri, Vittorio di Giorgio 


## Prerequisites
Install OpenSlide tools:
```sh
sudo apt-get install openslide-tools
pip-install -r requirements.txt #inside EFMI
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


## Run the baseline
To test one of the baseline latents with an svm classifier. Run ```EFMI/launch_scripts/test-baseline.sh root_dir num_images batch_size model_name latent_dim experiment_name```. The ```root_dir``` param is the dataset root directory containing the patches inside two distinct folders, in ROI and not in ROI. ```num_images``` (defaults to 24) specifies on how many images-patches to train the baseline, use this for test purposes (note that patches dirs contains image_name.svs/patch.png list). ```batch_size``` is the batch_size (defaults to 16). ```model_name``` specifies which pretrained baseline model to use as baseline feature extractor, defaults to resnet50. Availables: resnet50, densenet121. ```latent_dim``` specifies the extracted feature dimensionality (latent dimensions, defaults to 128). ```results_path``` specifies the file path in which to save experiment results.



# Feature extractors
## VAE
To test the custom VAE latents with the svm classifier, simply run ```EFMI/launch_scripts/test-vae.sh root_dir num_images batch_size latent_dim num_epochs vae_type```. ```root_dir``` is the dataset root directory containing the patches inside two distinct folders, in ROI and not in ROI. ```num_images``` (defaults to 24) specifies on how many images-patches to train the VAE, use this for test purposes (note that patches dirs contains image_name.svs/patch.png list). ```batch_size``` is the batch_size with which to train the VAE (defaults to 16), ```latent_dim``` specifies the extract latents dimension (defaults to 100). Use ```vae_type``` to specify which flavor of VAE to use (availables: vae, resvae).

## PathDINO
To test PathDino latents with the svm classifier, simply run ```EFMI/launch_scripts/test-pathdino.sh root_dir num_images batch_size latent_dim fine_tune num_epochs```. ```root_dir``` is the dataset root directory containing the patches inside two distinct folders, in ROI and not in ROI. ```num_images``` (defaults to 24) specifies on how many images-patches to train the model, use this for test purposes (note that patches dirs contains image_name.svs/patch.png list). ```batch_size``` is the batch_size (defaults to 16), ```latent_dim``` specifies the extract latents dimension (defaults to 100). If you want to finetune the model, specify ```finetune``` running the script, otherwise it will default to ```pretrained```.  

## BYOL
...