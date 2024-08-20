from torch.utils.data import Dataset, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split

import os
import numpy as np
from PIL import Image
import re


# TODO: write a "load" function instead of doing stuff in init
class PatchedDatasetAugmented(Dataset):
    def __init__(
        self,
        root_dir,
        not_roi_path="not_roi_patches",
        in_roi_path="in_roi_patches",
        num_images=24,
        size = 512,
        transform=None,
        eval = False,
    ):
        self.root_dir = root_dir
        self.not_roi_path = not_roi_path
        self.in_roi_path = in_roi_path
        self.num_images = num_images
        self.image_paths = []
        self.labels = []
        self.patients = []
        self.coordinates = []
        self.eval = eval

        # Process not_roi_patches (label 0)
        not_roi_dir = os.path.join(root_dir, self.not_roi_path)

        for patient_id in range(1, self.num_images + 1):
            patient_dir = str(patient_id) + ".svs"
            patient_dir = os.path.join(not_roi_dir, str(patient_dir))

            for img_name in [f for f in os.listdir(patient_dir) if f.endswith(".png")]:
                self.image_paths.append(os.path.join(patient_dir, img_name))
                self.labels.append(0)
                self.patients.append(patient_id)
                self.coordinates.append(self._extract_coordinates(img_name))

        # Process in_roi_patches (label 1)
        in_roi_dir = os.path.join(root_dir, self.in_roi_path)
        for patient_id in range(1, self.num_images + 1):
            if patient_id != 21:
                patient_dir = str(patient_id) + ".svs"
                patient_dir = os.path.join(in_roi_dir, str(patient_dir))

                for img_name in [
                    f for f in os.listdir(patient_dir) if f.endswith(".png")
                ]:
                    self.image_paths.append(os.path.join(patient_dir, img_name))
                    self.labels.append(1)
                    self.patients.append(patient_id)
                    self.coordinates.append(self._extract_coordinates(img_name))
        image_size  = size
        p_blur = 0.5
        image_size = 256
        p_blur = 0.5
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
           
        ])

        
    def _extract_coordinates(self, img_name):
        # Extract x and y from the filename

        match = re.match(r"(\d+)_(\d+)_.*\.png", img_name)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            return (x, y)
        else:
            raise ValueError(
                f"Filename {img_name} does not match the expected pattern."
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGBA")  # Ensure image is RGBA
        image = np.array(image)[:, :, :3]  # Drop the alpha channel
        image = Image.fromarray(image)  # Convert back to PIL image
        label = self.labels[idx]
        patient_id = self.patients[idx]
        coordinates = self.coordinates[idx]
        image = self.transform(image)
        if self.eval:
            return image, label
        return image  


def train_test_split_dataset(dataset, test_size=0.2, random_state=None):
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=dataset.labels
    )
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    return train_dataset, test_dataset

