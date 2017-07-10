"""
File list Dataset Preparation
"""

import torch.utils.data as data
from PIL import Image
import SimpleITK as sitk
import numpy as np
import os
import pdb
#import os.path

def default_loader(path):
    dicts = path.split('.')
    if dicts[-1]=='bmp':
        tar = Image.open(path).convert('P')
        return tar
    else:
        #img = np.squeeze((np.clip(sitk.GetArrayFromImage(sitk.ReadImage(path)) + 1024, 0, 2674) / 2674).astype('uint8'))
        #img = np.squeeze((np.clip(sitk.GetArrayFromImage(sitk.ReadImage(path)) + 1024, 0, 2674))).astype('float')
        img = np.squeeze((np.clip(sitk.GetArrayFromImage(sitk.ReadImage(path)) + 1024, 0, 2674))).astype('float') / 2674 * 255
        return Image.fromarray(img).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label (same to caffe's filelist)
    """
    filelist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, tarpath = line.strip().split()
            filelist.append((impath, tarpath))
    return filelist


class BoneTumor_Dataset(data.Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None, pair_transform=None,
                     flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.filelist = flist_reader(os.path.join(root, flist))
        self.transform = transform
        self.target_transform = target_transform
        self.pair_transform = pair_transform
        self.loader = loader


    def __getitem__(self, index):
        impath, tarpath = self.filelist[index]
        img = self.loader(os.path.join(self.root, impath))
        tar = self.loader(os.path.join(self.root, tarpath))

        if self.pair_transform is not None:
            img, tar = self.pair_transform(img, tar)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            tar = self.target_transform(tar)

        return img, tar


    def __len__(self):
        return len(self.filelist)
