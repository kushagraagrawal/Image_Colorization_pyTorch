import torch
import os, time
from basic_model import Net
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from colorize_data import ColorizeData, getTrainValData, make_dataloaders,AverageMeter, visualize_image, save_checkpoint
import argparse
from PIL import Image

parser = argparse.ArgumentParser(description='Training and Using ResNet')
parser.add_argument('--data', default='../landscape_images/',metavar='DIR', help='path to dataset')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 0)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to .pth file checkpoint (default: none)')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (overridden if loading from checkpoint)')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='size of mini-batch (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='learning rate at start of training')
parser.add_argument('--weight-decay', '--wd', default=1e-10, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='use this flag to validate without training')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--inference', '-i', default=False, type=bool, metavar='N', help='perform inference')
parser.add_argument('--infercheckpoint', default='checkpoints/checkpoint-epoch-0.pth.tar', type=str, metavar='N', help='saved model checkpoint')
parser.add_argument('--inferimage', default='outputs/gray/img-0-epoch-0.jpg', type=str, metavar='N', help='image to be inferred')

class Trainer:
    def __init__(self):
        self.train_dataset = getTrainValData(args.data, mode='Train')
        self.train_dataloader = make_dataloaders(dataset=self.train_dataset, batch_size=args.batch_size, n_workers=args.workers)
        self.val_dataset = getTrainValData(args.data, mode='Val')
        self.val_dataloader = make_dataloaders(dataset=self.val_dataset, batch_size=args.batch_size, n_workers=args.workers)
        self.model = Net()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # Define hparams here or load them from a config file
    def train(self, epoch, use_gpu):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        self.model.train()

        end = time.time()

        for i, (input_gray, target) in enumerate(self.train_dataloader):

            if(use_gpu):
                input_gray = input_gray.cuda()
                target = target.cuda()

            data_time.update(time.time() - end)

            output = self.model(input_gray)
            loss = self.criterion(output, target)
            
            losses.update(loss.data, input_gray.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        epoch, i, len(self.train_dataloader), batch_time=batch_time,
                    data_time=data_time, loss=losses)) 
        
        print('Finished training epoch {}'.format(epoch))
        


    def validate(self,save_images, epoch, use_gpu):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
    
        # Switch model to validation mode
        self.model.eval()
    
        # Run through validation set
        end = time.time()
        for i, (input_gray, target) in enumerate(self.val_dataloader):

            if(use_gpu):
                input_gray = input_gray.cuda()
                target = target.cuda()
        
            
            # Record time to load data (above)
            data_time.update(time.time() - end)
            output = self.model(input_gray) # throw away class predictions
            loss = self.criterion(output, target) # check this!
            
            # Record loss and measure accuracy
            losses.update(loss.data, input_gray.size(0))

            # Save images to file
            if save_images:
                for j in range(len(output)):
                    save_path = {'grayscale': 'outputs/gray/', 'colorized': 'outputs/color/'}
                    save_name = 'img-{}-epoch-{}.jpg'.format(i * self.val_dataloader.batch_size + j, epoch)
                    visualize_image(input_gray[j], ab_input=output[j].data, show_image=False, save_path=save_path, save_name=save_name)

            # Record time to do forward passes and save images
            batch_time.update(time.time() - end)
            end = time.time()
        
            # Print model accuracy -- in the code below, val refers to both value and validation
            if i % args.print_freq == 0:
                print('Validate: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        i, len(self.val_dataloader), batch_time=batch_time, loss=losses))

            print('Finished validation.')
            return losses.avg

best_losses = 1000.0
use_gpu = torch.cuda.is_available()
def main():
    global args, best_losses
    args = parser.parse_args()
    print('Arguments: {}'.format(args))
    if(args.inference == False):
        trainer = Trainer()

    if(use_gpu):
        trainer.model.cuda()
        trainer.criterion.cuda()
        print('Loaded model onto GPU')

    if(args.resume):
        if os.path.isfile(args.resume):
            print('Loading checkpoint {}...'.format(args.resume))
            checkpoint = torch.load(args.resume) if use_gpu else torch.load(args.resume, map_location=lambda storage, loc: storage)
            args.start_epoch = checkpoint['epoch']
            best_losses = checkpoint['best_losses']
            trainer.model.load_state_dict(checkpoint['state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer'])
            print('Finished loading checkpoint')

    if(args.inference == False):
        trainer.validate(False, 0, use_gpu=use_gpu)
        for epoch in range(args.start_epoch, args.epochs):
            trainer.train(epoch, use_gpu=use_gpu)
            save_images = True
            losses = trainer.validate(save_images, epoch, use_gpu=use_gpu)

            is_best_so_far = losses < best_losses
            best_losses = max(losses, best_losses)
            save_checkpoint({
                'epoch': epoch + 1,
                'best_losses': best_losses,
                'state_dict': trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
            }, is_best_so_far, 'checkpoints/checkpoint-epoch-{}.pth.tar'.format(epoch))
    else:
        inference(args.inferimage, args.infercheckpoint)

def inference(image, PATH):
    from torchvision.utils import save_image
    model = Net()
    model_checkpoint = torch.load(PATH, map_location=lambda storage, loc: storage)
    model.load_state_dict(model_checkpoint['state_dict'])
    model.eval()

    input_image = Image.open(image)
    infer_transform = transforms.Compose([transforms.Resize(size=(256,256)),
                                          transforms.Grayscale(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5), (0.5))
                                          ])
    input_image = infer_transform(input_image).float()
    input_image = input_image.unsqueeze(0)

    outputs = model(input_image)
    saveFileName = image.split('.')
    save_image(outputs, saveFileName[0] + '_inference.' + saveFileName[1])


if __name__ == '__main__':
    main()