import os
import argparse
import multiprocessing
from pathlib import Path
from PIL import Image


import torchvision.transforms as T

import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

from torch.utils.data import DataLoader, Dataset
import kornia.augmentation as K
import kornia
from byol_pytorch import BYOL
import pytorch_lightning as pl
from dataset.patches_dataset import PatchedDatasetAugmented, train_test_split_dataset
from byol_pytorch.trainer import BYOLTrainer
from byol_pytorch.trainer import MockDataset
# test model, a resnet 50x

resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# arguments

parser = argparse.ArgumentParser(description='byol-lightning-test')

parser.add_argument('--image_folder', type=str, required = True,
                       help='path to your folder of images for self-supervised learning')

parser.add_argument('--features', type=str, required = False, default="training",
                       help='')

args = parser.parse_args()

# constants
BATCH_SIZE = 32
EPOCHS     = 30
LR         = 3e-4
NUM_GPUS   = 2
IMAGE_SIZE = 256
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
NUM_WORKERS = multiprocessing.cpu_count()

# pytorch lightning module


# images dataset



# main

if __name__ == '__main__':
    print("Starting....")
    print(args.features)
   
   
    ds = PatchedDatasetAugmented(args.image_folder, size=IMAGE_SIZE)


    train_set, test_set = train_test_split_dataset(ds, test_size=0.2) 
    
    
    
    
    
    train_loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    p_blur = 0.5    
    augment_fn = torch.nn.Sequential(
    K.RandomHorizontalFlip()
)

    augment_fn2 = torch.nn.Sequential(
    K.RandomHorizontalFlip(),
    kornia.filters.GaussianBlur2d((3, 3), (1.5, 1.5))
)
    
        
    transform1 = torch.nn.Sequential(
    K.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.2, 1.0)),
    K.RandomHorizontalFlip(),
    K.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
    K.RandomGaussianBlur((IMAGE_SIZE // 20 * 2 + 1, IMAGE_SIZE // 20 * 2 + 1), (0.1, 2.0), p=0.5),
    K.RandomGrayscale(p=0.2),
    K.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
)

    transform2 = torch.nn.Sequential(
    K.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.2, 1.0)),
    K.RandomHorizontalFlip(),
    K.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
    K.RandomGaussianBlur((IMAGE_SIZE // 20 * 2 + 1, IMAGE_SIZE // 20 * 2 + 1), (0.1, 2.0), p=0.5),
    K.RandomGrayscale(p=0.2),
    K.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
)

    byol_args = {
       
     
        'projection_size': 256,
        'projection_hidden_size': 4096,
        'moving_average_decay': 0.99,
        'augment_fn': transform1,
        'augment_fn2': transform2,

       
    }


   
    trainer = BYOLTrainer(
    resnet,
    dataset = train_set,
    image_size = IMAGE_SIZE,
    hidden_layer = 'avgpool',
    learning_rate = 3e-4,
    num_train_steps = 30,
    batch_size = BATCH_SIZE,
    checkpoint_every = 100,
    checkpoint_folder = './checkpoints',
    resume_from_checkpoint = True,
    byol_kwargs = byol_args,

    )

    trainer()







