

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import pickle
from tqdm import tqdm
from mtcnn import MTCNN


class FaceDataset(Dataset):

    def __init__(self, image_dir, num_images=7000, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
                            if fname.endswith(('.jpg', '.png', '.jpeg'))]

        if len(self.image_paths) > num_images:
            self.image_paths = random.sample(self.image_paths, num_images)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


class FaceDatasetWithWeightMaps(Dataset):
    """Custom Dataset with pre-generated MTCNN weight maps"""

    def __init__(self, image_dir, weight_maps, num_images=7000, transform=None):
        self.image_dir = image_dir
        self.weight_maps = weight_maps
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
                            if fname.endswith(('.jpg', '.png', '.jpeg'))]

        if len(self.image_paths) > num_images:
            self.image_paths = random.sample(self.image_paths, num_images)
        
        if len(self.weight_maps) < len(self.image_paths):
            print(f"Warning: Need {len(self.image_paths)} weight maps, but only have {len(self.weight_maps)}")
            self.image_paths = self.image_paths[:len(self.weight_maps)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        weight_map = torch.from_numpy(self.weight_maps[idx]).float()
        
        return image, weight_map
