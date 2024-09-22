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


def Custom_loss(outputs, labels, criterion, means, lamb):
    T = outputs.size(1)
    Loss_mmd = 0
    Loss_es = criterion(torch.mean(outputs,dim=1), labels)
    if lamb != 0:
        for t in range(T):
            MMDLoss = torch.nn.LogSoftmax(dim=1)
            targets = F.softmax(torch.mean(outputs,dim=1)/means, dim =1)
            log_probs = MMDLoss(outputs[:, t, ...])
            Loss_mmd += means * means * (-targets * log_probs).mean(0).sum()
    else:
        Loss_mmd = 0
    return (1 - lamb) * Loss_es + lamb *  Loss_mmd / T

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

### ACC计算工具
# eg: top1 = AverageMeter
# prec1 = accuracy(output.data, target)[0]
# top1.update(prec1.item(), input.size(0))
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train_epoch(args,epoch, data_loader, net, criterion, optimizer, device, schedula,part, tb_writer = None, wb_writer = None,
scaler=None):
    net.train()
    #batch_time = AverageMeter()
    losses = AverageMeter()
    losses2 = AverageMeter()
    
    top1 = AverageMeter()
    s_time = time.time()
    r = np.random.rand(1)

    for i, (images, labels) in enumerate(data_loader):
        labels = labels.to(device)
        images = images.to(device)
        if scaler is not None:
            with amp.autocast():
                out1 = net(images)
                out = torch.mean(out1,dim=1)
                if not opt.custom:
                    loss = Custom_loss(out1, labels, criterion, opt.means,part)
                else:
                    # compute output
                    outputs = net(images)
                    out = torch.mean(outputs,dim=1)
                    loss = criterion(out, labels)
                
        else:
            outputs = net(images)
            out = torch.mean(outputs,dim=1)
            loss = criterion(out, labels)
            
            
            

        optimizer.zero_grad()

        if scaler is not None:
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            optimizer.step()
        prec1 = accuracy(out.data, labels)[0]
        losses.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))

        if (i + 1) % (len(train_dataset) // (opt.batch_size * 5)) == 0 :
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f, Time: %.2f' % (epoch +1, opt.epoch, i+1, len(train_dataset) // opt.batch_size, losses.avg, time.time() - s_time))
            s_time = time.time()
    if tb_writer is not None:
        tb_writer.add_scalar('train/loss', losses.avg, epoch+1)
        tb_writer.add_scalar('train/accuracy', top1.avg, epoch+1)
    if wb_writer is not None:
        ws1.write(epoch+1,0,epoch)
        ws1.write(epoch+1,1,top1.avg)
        ws2.write(epoch+1,0,epoch)
        ws2.write(epoch+1,1,losses.avg)

        
    print('Train Accuracy: %.2f%%' %(top1.avg))
    train_scores.append(top1.avg)
    schedula.step()
        



# test
def test_epoch(args,epoch, data_loader, net, device, tb_writer = None, wb_writer = None):

    net.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            out = net(inputs)
            out = torch.mean(out, dim=1)
            pred = out.max(1)[1]
            total += targets.size(0)
            correct += (pred ==targets).sum()
        acc = 100.0 * correct.item() / total
        
        print('Test correct: %d Accuracy: %.2f%%' % (correct, acc))
        if tb_writer is not None:
            tb_writer.add_scalar('test/accuracy', acc, epoch+1)
            
        if wb_writer is not None:

            ws3.write(epoch+1,0,epoch)
            ws3.write(epoch+1,1,acc)
        

        test_scores.append(acc)

        if acc >=max(test_scores):
            save_file = str('best.pth')
            torch.save(net.state_dict(), os.path.join(save_path, save_file))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--epoch', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--dts', type=str, default='CIFAR10')
    parser.add_argument('--model', type=str, default='MSNN')
    parser.add_argument('--result_path', type = Path, default = './result' )
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--custom', type=bool, default=False)
    parser.add_argument('--Temperature', type=float, default=1)
    parser.add_argument('--lamb', type=float, default=0.0)
    parser.add_argument('--amp', type=bool,default=True)


    
    
    opt = parser.parse_args()

    torch.cuda.set_device(opt.gpu)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True


    save_path = './' + opt.model + '_' + opt.dts + '_' + str(opt.seed)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        


# for cifar10
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    Cutout(n_holes=1, length=16)])
# Cutout(n_holes=1, length=16)
    transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        
        
    if opt.dts == 'CIFAR10':
        train_dataset = dsets.CIFAR10(root='', train=True, transform=transform_train, download=True)
        test_dataset = dsets.CIFAR10(root='', train=False,  transform=transform_test)
    elif opt.dts == 'CIFAR100':
        train_dataset = dsets.CIFAR100(root='', train=True, transform=transform_train, download=True)
        test_dataset = dsets.CIFAR100(root='', train=False, transform=transform_test)
    elif opt.dts == 'MNIST':
        train_dataset = dsets.MNIST(root='', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = dsets.MNIST(root='', train=False,
                                     transform=transforms.ToTensor())
    elif opt.dts == 'Fashion-MNIST':
        train_dataset = dsets.FashionMNIST(root = '', train = True, transform = transforms.ToTensor(), download = True)
        test_dataset = dsets.FashionMNIST(root = '', train = False, transform = transforms.ToTensor())
    else:
        pass

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size,  pin_memory = True,shuffle=True,num_workers = 1)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size, pin_memory = True,shuffle=False,num_workers = 1)
    model = resnet19()
    # model = resnet20()
    model.T = opt.T
    #model.to(device)
    model.to(device)
    if opt.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None
    test_scores = []
    train_scores = []

    optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=5e-5)
    # optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=5e-4)
    loss_function = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate,momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[40,60],gamma = 0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=opt.epoch)
    tb_writer = SummaryWriter(log_dir= './result')
    
    wb_writer = xlwt.Workbook()
    ws1 = wb_writer.add_sheet('train_accuracy')
    ws2 = wb_writer.add_sheet('train_loss')
    ws3 = wb_writer.add_sheet('test_accuracy')
    
    ws1.write(0,0,'Epoch')
    ws1.write(0,1,'ACC')
    ws2.write(0,0,'Epoch')
    ws2.write(0,1,'Loss')
    ws3.write(0,0,'Epoch')
    ws3.write(0,1,'ACC')

    
    for epoch in range(opt.epoch): 
        # ppp = calc_n(0.6,epoch,opt.epoch)
        # ppp = 0
    
        start_time = time.time()
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))

        train_epoch(opt,epoch, train_loader, model, loss_function, optimizer, device, scheduler,ppp, tb_writer,wb_writer,scaler)
        
        if (epoch + 1) % 1 == 0:
            test_epoch(opt,epoch,  test_loader, model, device, tb_writer,wb_writer)
            print(f'escape time={(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (opt.epoch - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n') 
        if (epoch + 1) % 20 ==0:
            print('Best Test Accuracy in %d: %.2f%%' % (epoch + 1, max(test_scores)))
            print('Best Train Accuracy in %d: %.2f%%' % (epoch + 1, max(train_scores)))
        if epoch+1 ==opt.epoch:
            print('Best Test Accuracy in %d: %.2f%%' % (epoch + 1, max(test_scores)))
            print('Best Train Accuracy in %d: %.2f%%' % (epoch + 1, max(train_scores)))
        wb_writer.save('./MSNN_CIFAR10_2023/2024-6-20_23.xls')