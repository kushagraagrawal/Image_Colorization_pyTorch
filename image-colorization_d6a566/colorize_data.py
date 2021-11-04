from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import datasets
import torch
import numpy as np
import os
from PIL import Image
import glob
import matplotlib.pyplot as plt
from skimage.color import lab2rgb, rgb2lab, rgb2gray
import shutil

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
        self.target_transform = T.Compose([T.Resize(size=(256,256)),
                                           T.ToTensor(),
                                           T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        self.paths = paths
    
    def __len__(self) -> int:
        # return Length of dataset
        return len(self.paths)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return the input tensor and output tensor for training
        input_img = Image.open(self.paths[index]).convert("RGB")
        transformed_img = self.input_transform(input_img)
        target = self.target_transform(input_img)
        return (transformed_img, target)

def make_dataloaders( dataset:Dataset, batch_size=16, n_workers=4, pin_memory=True):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, pin_memory=pin_memory)
    return dataloader

def getTrainValData(directory, mode='Train'):
    paths = glob.glob(directory + '*.jpg')
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


class AverageMeter(object):
    '''A handy class from the PyTorch ImageNet tutorial''' 
    def __init__(self):
        self.reset()
    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def visualize_image(grayscale_input, ab_input=None, show_image=False, save_path=None, save_name=None):
    '''Show or save image given grayscale (and ab color) inputs. Input save_path in the form {'grayscale': '/path/', 'colorized': '/path/'}'''
    plt.clf() # clear matplotlib plot
    ab_input = ab_input.cpu()
    grayscale_input = grayscale_input.cpu()    
    if ab_input is None:
        grayscale_input = grayscale_input.squeeze().numpy() 
        if save_path is not None and save_name is not None: 
            plt.imsave(grayscale_input, '{}.{}'.format(save_path['grayscale'], save_name) , cmap='gray')
        if show_image: 
            plt.imshow(grayscale_input, cmap='gray')
            plt.show()
    else:
        ab_input[0] = grayscale_input[0]
        color_image = ab_input.numpy()
        color_image = color_image.transpose((1, 2, 0))  
        color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
        color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   
        color_image = lab2rgb(color_image.astype(np.float64))
        grayscale_input = grayscale_input.squeeze().numpy()
        if save_path is not None and save_name is not None:
            plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
            plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))
        if show_image: 
            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(grayscale_input, cmap='gray')
            axarr[1].imshow(color_image)
            plt.show()

def save_checkpoint(state, is_best_so_far, filename='checkpoints/checkpoint.pth.tar'):
    '''Saves checkpoint, and replace the old best model if the current model is better'''
    torch.save(state, filename)
    if is_best_so_far:
        shutil.copyfile(filename, 'checkpoints/model_best.pth.tar')

# train_dl = make_dataloaders(paths=train_paths)
# val_dl = make_dataloaders(paths=val_paths)