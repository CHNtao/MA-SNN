import argparse
import shutil
import os
import time
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
from resnet_models import *
from pathlib import Path
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import datetime
import xlwt
from layers import *
from torch.cuda import amp
import pandas as pd

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Altered from https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    Args:
    n_holes (int): Number of patches to cut out of each image.
    length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
        img (Tensor): Tensor image of size (C, H, W).
        Returns:
        Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img = img * mask

        return img

parser = argparse.ArgumentParser(description='main.py')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=64)
opt = parser.parse_args()

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.cuda.set_device(opt.gpu)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                Cutout(n_holes=1, length=16)])
transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
# to do: set the dataset path    
train_dataset = dsets.CIFAR10(root='/data_dir', train=True, transform=transform_train, download=True)
test_dataset = dsets.CIFAR10(root='/data_dir', train=False, transform=transform_test)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size,  pin_memory = True,shuffle=True,num_workers = 1)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size, pin_memory = True,shuffle=False,num_workers = 1)
# to do: set the trained model path
pretrained_model_1 ='/weight_dir'

# model1 
model1 = resnet19()
model1.T = 4
model1.to(device)
model1.load_state_dict(torch.load(pretrained_model_1,map_location=torch.device('cpu')))
model1.eval()


total = 0
correct1 = 0
correct2 = 0
with torch.no_grad():
    for i, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        output1 = model1(inputs)
        out1 = torch.mean(output1, dim=1)
        pred1 = out1.max(1)[1]
        total += targets.size(0)
        correct1 += (pred1 ==targets).sum()
    acc1 = 100.0 * correct1.item() / total
    print('Test correct1: %d  Accuracy_baseline: %.2f%% ' % (correct1, acc1))






