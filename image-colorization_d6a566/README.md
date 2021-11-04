# Image Colorization Starter Code
The objective is to produce color images given grayscale input image. 

## Setup Instructions
Create a conda environment with pytorch, cuda. 

`$ conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia`

For systems without a dedicated gpu, you may use a CPU version of pytorch.
`$ conda install pytorch torchvision torchaudio cpuonly -c pytorch`

## Dataset
Use the zipfile provided as your dataset. You are expected to split your dataset to create a validation set for initial testing. Your final model can use the entire dataset for training. Note that this model will be evaluated on a test dataset not visible to you.

## Code Guide
Baseline Model: A baseline model is available in `basic_model.py` You may use this model to kickstart this assignment. We use 256 x 256 size images for this problem.
-	Fill in the dataloader, (colorize_data.py)
-	Fill in the loss function and optimizer. (train.py)
-	Complete the training loop, validation loop (train.py)
-	Determine model performance using appropriate metric. Describe your metric and why the metric works for this model? 
- Prepare an inference script that takes as input grayscale image, model path and produces a color image. 

## Additional Tasks 
- The network available in model.py is a very simple network. How would you improve the overall image quality for the above system? (Implement)
- You may also explore different loss functions here.

## Bonus
You are tasked to control the average color/mood of the image that you are colorizing. What are some ideas that come to your mind? (Bonus: Implement)

## Solution
- Document the things you tried, what worked and did not. 
- Update this README.md file to add instructions on how to run your code. (train, inference). 
- Once you are done, zip the code, upload your solution.  

## Instructions
- Use the below command to train the network from scratch
`python train.py $path_to_image_dir$ --epochs $Num_of_epochs$`
- Please use the below argument parsers commands to include required hyperparameters
```
usage: train.py [-h] [-j N] [--resume PATH] [--epochs N] [--start-epoch N]
                [-b N] [--lr LR] [--weight-decay W] [-e] [--print-freq N]
                [--inference N] [--infercheckpoint N] [--inferimage N]
                DIR

Training and Using ResNet

optional arguments:
  -h, --help            show this help message and exit
  --data DIR            path to dataset
  -j N, --workers N     number of data loading workers (default: 0)
  --resume PATH         path to .pth file checkpoint (default: none)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (overridden if loading from
                        checkpoint)
  -b N, --batch-size N  size of mini-batch (default: 16)
  --lr LR, --learning-rate LR
                        learning rate at start of training
  --weight-decay W, --wd W
                        weight decay (default: 1e-4)
  -e, --evaluate        use this flag to validate without training
  --print-freq N, -p N  print frequency (default: 10)
  --inference N, -i N   perform inference
  --infercheckpoint N   saved model checkpoint
  --inferimage N        image to be inferred

```
- After training is complete, you can run train.py to run inference on a single image as below <br>
`python train.py --inference True inferimage $PATH_TO_IMAGE$` <br>
It dumps an image in the same path with the suffix _inference
