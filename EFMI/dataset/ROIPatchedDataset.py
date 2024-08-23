import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import openslide
import cv2 as cv
import xml.etree.cElementTree as ET
from typing import List, Tuple


class ROIPatchDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        wsi_paths: List[str],
        xml_paths: List[str],
        patch_size: int = 512,
        mag_level: int = 1,
        transform=transforms.Compose([transforms.ToTensor()]),
    ):
        self.root_dir = root_dir
        self.wsi_paths = wsi_paths
        self.xml_paths = xml_paths
        self.patch_size = patch_size
        self.mag_level = mag_level
        self.transform = transform
        self.patches = []
        self.roi_coords = []

        self._build_dataset()

    def _build_dataset(self):
        for wsi_path, xml_path in zip(self.wsi_paths, self.xml_paths):
            slide = openslide.open_slide(wsi_path)
            roi_coords = self._parse_xml(xml_path)

            width, height = slide.level_dimensions[self.mag_level]
            downsampling_factor = slide.level_downsamples[self.mag_level]

            for y in range(0, height - self.patch_size, self.patch_size):
                for x in range(0, width - self.patch_size, self.patch_size):
                    patch = slide.read_region(
                        (int(x * downsampling_factor), int(y * downsampling_factor)),
                        self.mag_level,
                        (self.patch_size, self.patch_size),
                    )

                    if self._is_valid_patch(patch):
                        patch_roi = self._get_patch_roi(x, y, roi_coords)
                        self.patches.append(patch)
                        self.roi_coords.append(patch_roi)

    def parse_xml(xml_path):
        """
        builds the list that represent the ROI path starting from an xml annots file
        """
        tree = ET.ElementTree(file=xml_path)
        annolist = []
        root = tree.getroot()
        for coords in root.iter("Vertices"):
            for coord in coords:
                x = int(float(coord.attrib.get("X")))
                y = int(float(coord.attrib.get("Y")))
                annolist.append((x, y))
        return annolist

    def _is_valid_patch(
        self, patch: Image.Image, threshold=0.95, gray_threshold=200
    ) -> bool:
        """
        Check if the patch is not predominantly white or light gray.
        """
        gray = cv.cvtColor(np.array(patch), cv.COLOR_RGB2GRAY)
        _, binary = cv.threshold(gray, gray_threshold, 255, cv.THRESH_BINARY)
        foreground_pixels = cv.countNonZero(binary)
        foreground_ratio = foreground_pixels / (patch.shape[0] * patch.shape[1])
        return foreground_ratio < threshold

    def _get_patch_roi(
        self, x: int, y: int, roi_coords: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        patch_roi = []
        for rx, ry in roi_coords:
            if x <= rx < x + self.patch_size and y <= ry < y + self.patch_size:
                patch_roi.append((rx - x, ry - y))
        return patch_roi

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        roi = self.roi_coords[idx]

        if self.transform:
            patch = self.transform(patch)

        return patch, torch.tensor(roi, dtype=torch.float32)
