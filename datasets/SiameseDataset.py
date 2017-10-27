"""
Siamese Dataset Preparation
"""

import torch.utils.data as data
from PIL import Image
import os

def default_loader(path):
    return Image.open(path).convert('RGB')

def default_flist_reader(flist):
    """
    flist format: impath1 impath2 label (same to caffe's filelist)
    """
    pairlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath1, impath2, target = line.strip().split()
            pairlist.append((impath1, impath2, int(target)))
    return pairlist


class SiameseDataset(data.Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None, 
                     flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.pairlist = flist_reader(os.path.join(root, flist))
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
            
    def __getitem__(self, index):
        impath1, impath2, target = self.pairlist[index]
        img1 = self.loader(os.path.join(self.root, impath1))
        img2 = self.loader(os.path.join(self.root, impath2))
        
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img1, img2, target
        
    def __len__(self):
        return len(self.pairlist)

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
#         SiameseDataset(root="/data/datasets/SiameseFusion/", flist="list.txt",
#            transform=transforms.Compose([
#                transforms.RandomHorizontalFlip(),
#                transforms.ToTensor()
#         ])),
#        batch_size=128, shuffle=True,
#        num_workers=4, pin_memory=True)
#
#for img1, img2, target in train_loader:
#    img1, img2, target = img1.cuda(), img2.cuda(), target.cuda()
#    img1, img2, target = Variable(img1), Variable(img2), Variable(target) 
