"""
File list Dataset Preparation
"""

import torch.utils.data as data
from PIL import Image
import os
#import os.path

def default_loader(path):
    dicts = path.split('.')
    if dicts[-1]=='jpg':
        return Image.open(path).convert('RGB')
    else:
        return Image.open(path).convert('P')

def default_flist_reader(flist):
    """
    flist format: impath label (same to caffe's filelist)
    """
    filelist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            filelist.append(line.strip())
    return filelist


class VOC_Dataset(data.Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None, 
                     flist_reader=default_flist_reader, loader=default_loader):
        self.images_root = os.path.join(root, 'JPEGImages')
        self.targets_root = os.path.join(root, 'SegmentationClass')

        self.filelist = flist_reader(os.path.join(root, flist))
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
            
    def __getitem__(self, index):
        impath = self.filelist[index]
        img = self.loader(os.path.join(self.images_root, impath+'.jpg'))
        target = self.loader(os.path.join(self.targets_root, impath+'.png'))
        
        if self.transform is not None:
            img = self.transform(img) * 255
        
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target
        
    def __len__(self):
        return len(self.filelist)
