# ML-in-Apps-2024-AM03-Extracting-Features-from-Medical-Images
Repository of Politecnico di Torino Master's degree course Machine Learning in Applications 

## Prerequisites
Install OpenSlide tools:
```sh
sudo apt-get install openslide-tools
```

## Extract patches from WSI images
To extract annotated patches from wsi images, run ```EFMI/launch_scripts/extract-patches.sh DATASET_PATH NOT_ROI_PATH ROI_PATH PATCH_SIZE MAG_LEVEL```. This requires that ```DATASET_PATH``` contains both svs image files and xml annotation files. The mapping svs -> xml is given by their names, i.e ```1.svs -> 1.xml```



## Use the PatchedDataset
Example usage:

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
