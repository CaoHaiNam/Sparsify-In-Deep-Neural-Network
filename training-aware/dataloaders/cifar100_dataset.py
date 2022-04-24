import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from .cifar100_config import *
import numpy as np

def cifar100_train_loader(dataset_name, train_batch_size, num_workers=0, pin_memory=False, normalize=None):
    size = [3,32,32]
    if normalize is None:
        normalize = transforms.Normalize(
            mean=mean[dataset_name], std=std[dataset_name])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder('../dat/cifar100_org/train/{}'.format(dataset_name),
            train_transform)
    
    return torch.utils.data.DataLoader(train_dataset,
        batch_size=train_batch_size, shuffle=True, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)

    # loader = torch.utils.data.DataLoader(train_dataset,
    #     batch_size=train_batch_size, shuffle=False, sampler=None,
    #     num_workers=num_workers, pin_memory=pin_memory)
    # data = {'x': [],'y': []}
    # for image,target in loader:
    #     data['x'].append(image)
    #     data['y'].append(target.numpy()[0])

    # data['x']=torch.cat(data['x']).view(-1,size[0],size[1],size[2])
    # data['y']=torch.LongTensor(np.array(data['y'],dtype=int)).view(-1)
    # return data


def cifar100_val_loader(dataset_name, val_batch_size, num_workers=0, pin_memory=False, normalize=None):
    size = [3,32,32]
    if normalize is None:
        normalize = transforms.Normalize(
            mean=mean[dataset_name], std=std[dataset_name])

    val_dataset = \
        datasets.ImageFolder('../dat/cifar100_org/test/{}'.format(
                dataset_name),
                transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]))

    return torch.utils.data.DataLoader(val_dataset,
        batch_size=val_batch_size, shuffle=False, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)


    # loader = torch.utils.data.DataLoader(val_dataset,
    #     batch_size=val_batch_size, shuffle=False, sampler=None,
    #     num_workers=num_workers, pin_memory=pin_memory)

    # data = {'x': [],'y': []}
    # for image,target in loader:
    #     data['x'].append(image)
    #     data['y'].append(target.numpy()[0])

    # data['x']=torch.cat(data['x']).view(-1,size[0],size[1],size[2])
    # data['y']=torch.LongTensor(np.array(data['y'],dtype=int)).view(-1)
    # return data
