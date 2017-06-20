"""
Siamese network Implemented by ZhangYu

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Function

class mySiamese1S(nn.Module):
    """
        Define a siamese network with one input and Shared branch
        Given a module, it will duplicate it with weight sharing, concatenate the output and add a linear classifier 
    """
    # init the branches as net
    def __init__(self, net):
        super(mySiamese1S, self).__init__()
        self.features = net
    
    # calc the out put of each branch
    def forward_once(self, x):
        res = self.features(x)
        return res
    
    # forward function, input x should have twice of the original channels
    def forward(self, x):
        dims = x.data.size()
        x1, x2 = x.split(dim=0, split_size=dims[0]//2)
        
        o1 = self.forward_once(x1)
        o2 = self.forward_once(x2)
        
        dim1 = o1.data.size()[1]
        #dims2 = o2.data.size()
        o = Variable(torch.zeros(dim1*2))
        o[:dim1] = o1
        o[dim1:] = o2
        
        return o

class mySiamese1P(nn.Module):
    """
        Define a siamese network with one input and parallel branch
        Given a module, it will duplicate it with weight sharing, concatenate the output and add a linear classifier 
    """
    # init the branches as net
    def __init__(self, net):
        super(mySiamese1P, self).__init__()
        self.features1 = net
        self.features2 = net
    
    # forward function, input x should have twice of the original channels
    def forward(self, x):
        dims = x.data.size()
        x1, x2 = x.split(dim=0, split_size=dims[0]//2)
        
        o1 = self.features1(x1)
        o2 = self.features2(x2)
        
        dim1 = o1.data.size()[1]
        #dims2 = o2.data.size()
        
        o = Variable(torch.zeros(dim1*2))
        o[:dim1] = o1
        o[dim1:] = o2
        
        return o

class mySiamese2P(nn.Module):
    """
        Define a siamese network with two inputs and parallel branch
        Given a module, it will duplicate it with weight sharing, concatenate the output and add a linear classifier 
    """
    # init the branches as net
    def __init__(self, net):
        super(mySiamese2P, self).__init__()
        self.features1 = net
        self.features2 = net
    
    # forward function, input x should have twice of the original channels
    def forward(self, x1, x2):        
        o1 = self.features1(x1)
        o2 = self.features2(x2)
        
        dim1 = o1.data.size()[1]
        dim2 = o2.data.size()[1]
        
        o = Variable(torch.zeros(dim1+dim2))
        o[:dim1] = o1
        o[dim1:] = o2
        
        
        return o

class mySiamese2S(nn.Module):
    """
        Define a siamese network with two inputs and shared branch
        Given a module, it will duplicate it with weight sharing, concatenate the output and add a linear classifier 
    """
    # init the branches as net
    def __init__(self, net):
        super(mySiamese2S, self).__init__()
        self.features = net
    
    # calc the out put of each branch
    def forward_once(self, x):
        res = self.features(x)
        return res
    
    # forward function, input x should have twice of the original channels
    def forward(self, x1, x2):        
        o1 = self.forward_once(x1)
        o2 = self.forward_once(x2)
        
        dim1 = o1.data.size()[1]
        dim2 = o2.data.size()[1]
        o = Variable(torch.zeros(dim1+dim2))
        o[:dim1] = o1
        o[dim1:] = o2
        
        return o
