import os
import argparse
import multiprocessing
from pathlib import Path
from PIL import Image
from tqdm import tqdm


import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import kornia.augmentation as K

from byol_pytorch import BYOL
import pytorch_lightning as pl
from dataset.patches_dataset import PatchedDatasetAugmented, train_test_split_dataset

from byol_pytorch import BYOLTrainer
# test model, a resnet 50
net = models.resnet50(pretrained=True)
parser = argparse.ArgumentParser(description='byol-lightning-test')

parser.add_argument('--image_folder', type=str, required = True,
                       help='path to your folder of images for self-supervised learning')
args = parser.parse_args()

IMAGE_SIZE = 512
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
if __name__ == '__main__':

    byol_args = {
       'projection_size': 128,
       'projection_hidden_size': 4096,
       'moving_average_decay': 0.99,
      }


    byol = BYOL(net, image_size=IMAGE_SIZE, hidden_layer="avgpool", **byol_args)

    ds = PatchedDatasetAugmented(args.image_folder, size=IMAGE_SIZE, eval = True)
    dataloader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=4,)
    
    train_set, test_set = train_test_split_dataset(ds, test_size=0.2)
   
    checkpoint_folder = Path('./checkpoints')
    embeddings_with_labels = []
    checkpoints = list(checkpoint_folder.glob('checkpoint.*.pt'))
    if checkpoints:
            latest_checkpoint =max(checkpoints, key=lambda x: int(x.stem.split('.')[1]))
            print(f"Loading checkpoint from {latest_checkpoint}")
            
            checkpoint = torch.load(latest_checkpoint)
            
            new_state_dict = checkpoint['model_state_dict']
            print(new_state_dict.keys())

            net.load_state_dict(new_state_dict)
    net.eval()
       
    with torch.no_grad():
        for data, labels in tqdm(dataloader):
            projection, embedding = byol(data, return_embedding = True)
            embeddings_with_labels.append((embedding, labels))
    torch.save(embeddings_with_labels, './embeddings/eval_embeddings.pt')
    print("Embeddings saved")


    
        




