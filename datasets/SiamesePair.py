# coding: utf-8

# Siamese Dataset Preparation

import torch
import torch.utils.data as data
from PIL import Image

def default_loader(path):
    return Image.open(path).convert('RGB')

class SiamesePair(data.Dataset):
    def __init__(self, impath1, impath2, transform=None, target_transform=None, loader=default_loader):
        self.impath1 = impath1
        self.impath2 = impath2
        self.transform = transform
        self.loader = loader
            
    def get_pair(self):
        img1 = self.loader(self.impath1)
        img2 = self.loader(self.impath2)
        
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1, img2

    def get_source(self):
        img1 = self.loader(self.impath1)
        img2 = self.loader(self.impath2)

        return img1, img2
