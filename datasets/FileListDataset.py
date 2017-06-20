"""
File list Dataset Preparation
"""

import torch.utils.data as data
from PIL import Image
import os
#import os.path

def default_loader(path):
    return Image.open(path).convert('RGB')

def default_flist_reader(flist):
    """
    flist format: impath label (same to caffe's filelist)
    """
    filelist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, target = line.strip().split()
            filelist.append((impath, int(target)))
    return filelist


class FileListDataset(data.Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None, 
                     flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.filelist = flist_reader(os.path.join(root, flist))
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
            
    def __getitem__(self, index):
        impath, target = self.filelist[index]
        img = self.loader(os.path.join(self.root, impath))
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target
        
    def __len__(self):
        return len(self.filelist)

######################################################################
# """ Usage: train\_loader = torch.utils.data.DataLoader(
# ImageFilelist(root="../place365\_challenge/data\_256/",
# flist="../place365\_challenge/places365\_train\_challenge.txt",
# transform=transforms.Compose([transforms.RandomSizedCrop(224),
# transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,
# ])), batch\_size=64, shuffle=True, num\_workers=4, pin\_memory=True) """
# 
#
#train_loader = data.DataLoader(
#         FileListDataset(root="/data/datasets/SiameseFusion/", flist="list.txt",
#            transform=transforms.Compose([
#                transforms.RandomHorizontalFlip(),
#                transforms.ToTensor()
#         ])),
#        batch_size=128, shuffle=True,
#        num_workers=4, pin_memory=True)
#
#for img1, target in train_loader:
#    img1, target = img1.cuda(), target.cuda()
#    img1, target = Variable(img1), Variable(target) 
