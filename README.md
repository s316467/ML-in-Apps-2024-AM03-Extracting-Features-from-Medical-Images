# Extracting features from medical images
Authors: Maria Scoleri, Antonio Ferrigno, Fabio Barbieri, Vittorio di Giorgio 

# Project structure

In the following, you can find a brief description of the project files.

| File | Description | 
| ---- | ----------- |
| `launch_scripts/.*.sh` | contains the experiments run scripts |
| `results/.*` | contains experiments classification reports and plots |
| `dataset/PatchedDataset.py` | contains the custom Dataset class |
| `dataset/patches.py` | contains the code to extract patches from WSIs |
| `classifier/svm.py` | contains the svm classifier with which to test the extracted features on the main classification task (cancer presence) |
| `extractors/.*` | each subfolder contains the code to extract feature using a particular method |
| `extractors/baseline/.*` | contains the code to extract features from patches using imagenet pretrained baselines (resnet50, densenet121) |
| `extractors/baseline/features` | contains np arrays of baseline extracted features and labels |
| `extractors/pathdino/.*` |  contains the code to extract features from patches using pathdino (pretrained or finetuned) |
| `extractors/vae/.*` |  contains the code to extract features from patches using a custom VAE |




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
To test PathDino latents with the svm classifier, move in ```EFMI``` and run ```/launch_scripts/test-pathdino.sh root_dir num_images batch_size latent_dim fine_tune_epochs results_path pretrained_dino_path```. ```root_dir``` is the dataset root directory containing the patches inside two distinct folders, in ROI and not in ROI. ```num_images``` (defaults to 24, i.e: full dataset) specifies on how many images-patches to train the model, use this for test purposes (note that patches dirs contains image_name.svs/patch.png list). ```batch_size``` is the batch_size (defaults to 16), ```latent_dim``` specifies the extracted latents dimension (defaults to 128). If you want to finetune the model, specify a number for ```fine_tune_epochs``` running the script, otherwise it will default to ```0, i.e: without finetuning```. Specify where do you want to save the report with ```results_path``` (defaults to "./results/pathdino" and "./results/pathdino/finetune) and load pretrained weights from ```pretrained_dino_path``` (defaults to "./extractors/pathdino/model/PathDino512.pth)

## BYOL
...