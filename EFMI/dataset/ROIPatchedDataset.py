import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os


class ROIPatchDataset(Dataset):
    def __init__(
        self,
        patches_path="..",
        num_images=24,
        transform=transforms.Compose([transforms.ToTensor()]),
        image_paths=[],
        patches=[],
        rois=[],
    ):
        self.patches_path = patches_path
        self.transform = transform
        self.num_images = num_images
        self.image_paths = image_paths
        self.patches = patches
        self.rois = rois

        for patient_id in range(1, self.num_images + 1):
            patient_dir = str(patient_id) + ".svs"
            patient_dir = os.path.join(patches_path, str(patient_dir))

            for img_name in [f for f in os.listdir(patient_dir) if f.endswith(".png")]:
                self.image_paths.append(os.path.join(patient_dir, img_name))
            for roi_name in [f for f in os.listdir(patient_dir) if f.endswith(".npy")]:
                self.rois.append(np.load(os.path.join(patient_dir, roi_name)))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        roi = self.roi_coords[idx]

        if self.transform:
            patch = self.transform(patch)

        return patch, torch.tensor(roi, dtype=torch.float32)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGBA")  # Ensure image is RGBA
        image = np.array(image)[:, :, :3]  # Drop the alpha channel
        image = Image.fromarray(image)  # Convert back to PIL image
        roi = self.rois[idx]

        image = self.transform(image)
        roi = self.transform(roi)

        return image, roi
