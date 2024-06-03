import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import transforms
import os
import random

class CustomDatasetFolders(Dataset):
    def __init__(self, root_folders, transform=None):
        # Ensure root_folders is a list
        if isinstance(root_folders, str):
            root_folders = [root_folders]
        self.root_folders = root_folders
        self.transform = transform
        self.samples, self.labels = self._get_samples()
        print(f"CustomDatasetFolders: {len(self.samples)} samples found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        label = self.labels[index]
        image = default_loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    def _get_samples(self):
        samples = []
        labels = []
        categories = {'in_roi_patches': 0, 'not_roi_patches': 1}  # Assign numeric labels to each category
        root_folders = self.root_folders

        for root_folder in root_folders:
            for category, label in categories.items():
                category_path = os.path.join(root_folder, category)
                print(f"Looking for image files in: {category_path}")

                if not os.path.exists(category_path):
                    print(f"Warning: {category_path} does not exist.")
                    continue

                print(f"Processing folder: {category_path}")
                image_files = os.listdir(category_path)
                for image_file in image_files:
                    image_path = os.path.join(category_path, image_file)
                    print(f"Found image file: {image_path}")
                    samples.append(image_path)
                    labels.append(label)

        if not samples:
            print("No samples found!")
        else:
            print(f"Found {len(samples)} samples.")

        combined = list(zip(samples, labels))
        random.shuffle(combined)
        samples, labels = zip(*combined) if combined else ([], [])
        return list(samples), list(labels)


    
    """
    def _get_samples(self):
        samples = []
        categories = ['in_roi_patches', 'not_roi_patches']
        root_folders = self.root_folders
        
        for root_folder in root_folders:
            for category in categories:
                category_path = os.path.join(root_folder, category)
                print(f"Looking for svs folders in: {category_path}")
                
                for i in range(1, 25):
                    folder_name = f"{i}.svs"
                    folder_path = os.path.join(category_path, folder_name)
                    
                    if not os.path.exists(folder_path):
                        print(f"Warning: {folder_path} does not exist.")
                        continue
                    
                    print(f"Processing folder: {folder_path}")
                    image_files = os.listdir(folder_path)
                    for image_file in image_files:
                        image_path = os.path.join(folder_path, image_file)
                        samples.append(image_path)
                        
        random.shuffle(samples)
        return samples
    """
    """
    def _get_samples(self):
        samples = []
        root_folders = self.root_folders
        
        for root_folder in root_folders:
            print(f"Looking for PNG files in: {root_folder}")
            
            for i in range(1, 25):
                file_name = f"{i}.png"
                file_path = os.path.join(root_folder, file_name)
                
                if not os.path.exists(file_path):
                    print(f"Warning: {file_path} does not exist.")
                    continue
                
                print(f"Adding file: {file_path}")
                samples.append(file_path)
                            
        random.shuffle(samples)
        return samples
    """