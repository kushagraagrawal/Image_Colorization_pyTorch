import torch
import os


class Trainer:
    def __init__(self):
        pass
        # Define hparams here or load them from a config file
    def train(self):
        pass
        # dataloaders
        train_dataset = 
        train_dataloader = 
        val_dataset = 
        val_dataloader = 
        # Model
        model = 
        # Loss function to use
        criterion = 
        # You may also use a combination of more than one loss function 
        # or create your own.
        optimizer = 
        # train loop
        


    def validate(self):
        pass
        # Validation loop begin
        # ------
        # Validation loop end
        # ------
        # Determine your evaluation metrics on the validation dataset.