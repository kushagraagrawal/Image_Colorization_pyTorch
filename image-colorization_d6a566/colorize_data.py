from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch
import numpy as np
import os
from PIL import Image
import glob

from torchvision.transforms.functional import resize

class ColorizeData(Dataset):
    def __init__(self, paths):
        # Initialize dataset, you may use a second dataset for validation if required
        # Use the input transform to convert images to grayscale
        self.input_transform = T.Compose([T.Resize(size=(256,256)),
                                          T.Grayscale(),
                                          T.ToTensor(),
                                          T.Normalize((0.5), (0.5))
                                          ])
        # Use this on target images(colorful ones)
        self.target_transform = T.Compose([T.ToTensor(),
                                           T.Resize(size=(256,256)),
                                           T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        self.paths = paths
    
    def __len__(self) -> int:
        # return Length of dataset
        return len(self.paths)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return the input tensor and output tensor for training
        input_img = Image.open(self.paths[index])
        output_img = self.input_transform(input_img)
        return (input_img, output_img)

def make_dataloaders( dataset:Dataset, batch_size=16, n_workers=4, pin_memory=True):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, pin_memory=pin_memory)
    return dataloader

def getTrainValData(mode='Train'):
    paths = glob.glob('../landscape_images/*.jpg')
    np.random.seed(123)
    paths_subset = np.random.choice(paths, 4000, replace=False)
    rand_idxs = np.random.permutation(4000)
    train_idxs = rand_idxs[:3500]
    val_idxs = rand_idxs[3500:]
    train_paths = paths_subset[train_idxs]
    val_paths = paths_subset[val_idxs]
    if(mode == 'Train'):
        return ColorizeData(paths=train_paths)
    elif(mode == 'Val'):
        return ColorizeData(paths=val_paths)


# train_dl = make_dataloaders(paths=train_paths)
# val_dl = make_dataloaders(paths=val_paths)