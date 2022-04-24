import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import math
import copy

import utils
from utils import *
from arguments import get_args
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import importlib
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = get_args()
tstart = time.time()



print('=' * 100)
print('Arguments =')
for arg in vars(args):
    print('\t' + arg + ':', getattr(args, arg))
print('=' * 100)

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
else:
    print('[CUDA unavailable]'); sys.exit()

# dataloader = importlib.import_module('dataloaders.{}'.format(args.experiment))
if args.experiment == 'cifar10':
# -------- CIFAR10 --------
    train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='../dat/', train=True, download=True, transform=train_transforms
    )
    pruneset = torchvision.datasets.CIFAR10(
        root='../dat/', train=True, download=True, transform=test_transforms
    )
    testset = torchvision.datasets.CIFAR10(
        root='../dat/', train=False, download=True, transform=test_transforms
    )
    input_dim = [3, 32, 32]
    output_dim = 10

elif args.experiment == 'mnist':
# --------- MNIST ----------
    train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
    ])

    test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
    ])

    trainset = torchvision.datasets.MNIST(
        root='../dat/', train=True, download=True, transform=train_transforms
    )
    pruneset = torchvision.datasets.MNIST(
        root='../dat/', train=True, download=True, transform=train_transforms
    )
    testset = torchvision.datasets.MNIST(
        root='../dat/', train=False, download=True, transform=test_transforms
    )
    input_dim = [1, 28, 28]
    output_dim = 10

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, 
    drop_last=False, pin_memory=True,
)
 
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, pin_memory=True
)

prune_loader = torch.utils.data.DataLoader(pruneset, batch_size=args.batch_size, shuffle=True, 
                                                drop_last=False, pin_memory=True,)

        
approach = importlib.import_module('approaches.{}'.format(args.approach))


try:
    networks = importlib.import_module('networks.{}_net'.format(args.approach))
except:
    networks = importlib.import_module('networks.net')



Net = getattr(networks, args.arch)
# print(Net)


net = Net(input_dim, output_dim).cuda()


print(utils.print_model_report(net))

appr = approach.Appr(net, args=args)

utils.print_optimizer_config(appr.optimizer)
print('-' * 100)

if args.approach == 'sccl':
    appr.train(train_loader, test_loader, prune_loader)
else:
    appr.train(train_loader, test_loader)

t1 = time.time()
test_loss, test_acc = appr.eval(test_loader)
t2 = time.time()
print('>>> Test time {:5.1f}ms: loss={:.3f}, acc={:5.2f}% <<<'.format((t2-t1)*1000,test_loss, 100 * test_acc))