"""
Siamese network for image fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function

class myFusionNet1(nn.Module):
    """
        Define a siamese network with one input and Shared branch
        Given a module, it will duplicate it with weight sharing, concatenate the output and add a linear classifier 
    """
    # init the branches as net
    def __init__(self):
        super(myFusionNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.fc1 = nn.Linear(256*8*8*2, 256)
        self.fc2 = nn.Linear(256, 2)
    
    # calc the out put of each branch
    def forward_once(self, input):
        output = F.relu(self.conv1(input))
        output = F.max_pool2d(F.relu(self.conv2(output)), kernel_size=2,stride=2)
        output = F.relu(self.conv3(output))
        
        return output
    
    # forward function, input x should have twice of the original channels
    def forward(self, input):
        
        n_ch = input.data.size()[1]
        #print(input.data.size())
        #print(n_ch)
        input1 = input[:, :n_ch // 2, :, :]
        input2 = input[:, n_ch // 2 :, :, :]
        
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        
        output = torch.cat((output1, output2), 3)
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)
        
        return F.softmax(output)

class myFusionNet2(nn.Module):
    """
        Define a siamese network with one input and two branch
        Given a module, it will duplicate it with weight sharing, concatenate the output and add a linear classifier 
    """
    # init the branches as net
    def __init__(self):
        super(myFusionNet2, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1)
        
        self.conv1_2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1)
        
        self.fc1 = nn.Linear(256*8*8*2, 256)
        self.fc2 = nn.Linear(256, 2)
    
    # calc the out put of each branch
    def branch1_forward(self, input):
        output = F.relu(self.conv1_1(input))
        output = F.max_pool2d(F.relu(self.conv2_1(output)), kernel_size=2,stride=2)
        output = F.relu(self.conv3_1(output))
        
        return output
        
    def branch2_forward(self, input):
        output = F.relu(self.conv1_2(input))
        output = F.max_pool2d(F.relu(self.conv2_2(output)), kernel_size=2,stride=2)
        output = F.relu(self.conv3_2(output))
        
        return output
    
    # forward function, input x should have twice of the original channels
    def forward(self, input):
        # get channels
        n_ch = input.data.size()[1]
        #print(input.data.size())
        #print(n_ch)
        input1 = input[:, :n_ch // 2, :, :]
        input2 = input[:, n_ch // 2 :, :, :]
        
        output1 = self.branch1_forward(input1)
        output2 = self.branch2_forward(input2)
        
        output = torch.cat((output1, output2), 3)
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)
        
        return F.softmax(output)
