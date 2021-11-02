import torch
import os
from basic_model import Net
import torch.nn as nn
from torch import optim
from colorize_data import ColorizeData, getTrainValData, make_dataloaders


class Trainer:
    def __init__(self):
        pass
        # Define hparams here or load them from a config file
    def train(self):
        pass
        # dataloaders
        train_dataset = getTrainValData(mode='Train')
        train_dataloader = make_dataloaders(dataset=train_dataset)
        val_dataset = getTrainValData(mode='Val')
        val_dataloader = make_dataloaders(dataset=val_dataset)
        # Model
        model = Net()
        # Loss function to use
        criterion = nn.MSELoss()
        # You may also use a combination of more than one loss function 
        # or create your own.
        optimizer = optim.Adam()
        # train loop
        


    def validate(self):
        pass
        # Validation loop begin
        # ------
        # Validation loop end
        # ------
        # Determine your evaluation metrics on the validation dataset.