# coding: utf-8

# PyTorch Image Preparation

import torch
import torch.utils.data as data
from torchvision import datasets, transforms
from PIL import Image


def default_loader(path):
    return Image.open(path).convert('RGB')

class TorchImage(data.Dataset):
    def __init__(self, impath1, transform=None, target_transform=None, loader=default_loader):
        self.impath1 = impath1
        self.transform = transform
        self.loader = loader
            
    def get_image(self):
        img1 = self.loader(self.impath1)
        
        if self.transform is not None:
            img1 = self.transform(img1)
            
        return img1

    def get_source(self):
        img1 = self.loader(self.impath1)
        img2 = self.loader(self.impath2)

        return img1
