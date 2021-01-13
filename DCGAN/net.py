# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:57:15 2021

@author: hoon

Gan net's G &8 D
"""
import torch
from torch import nn
import numpy as np

class G(nn.Module):
    def __init__(self, in_channels=100, image_size=28):
        super(G, self).__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.main = nn.Sequential(
            self.convT_bn_relu(in_channels=self.in_channels, out_channels=image_size*8, k_size=7, stride=1, padding=0),
            self.convT_bn_relu(in_channels=self.image_size*8, out_channels=image_size*4, k_size=4, stride=2, padding=1),
            self.convT_bn_relu(in_channels=self.image_size*4, out_channels=1, k_size=4, stride=2, padding=1),
            )
        
    def convT_bn_relu(self, in_channels, out_channels, k_size=3, stride=1, padding=1):
        seq = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, k_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            )
        return seq
    
    def forward(self, x):
        x = x.view([-1,100,1,1])
        x = self.main(x)
        return x
    
class D(nn.Module):
    def __init__(self, in_channels=1, image_size=28):
        super(D, self).__init__()
        self.main = nn.Sequential(
            self.conv_bn_relu(in_channels, image_size*4, 3, 2, 1),
            self.conv_bn_relu(image_size*4, image_size*2, 3, 2, 1),
            self.conv_bn_relu(image_size*2, image_size*1, 3, 2, 1),
            self.conv_bn_relu(image_size*1, 1, 3, 2, 0),
            )
    def conv_bn_relu(self, in_channels, out_channels, k_size=3, stride=1, padding=1):
        seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, k_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            )
        return seq
    def forward(self, x):
        x = self.main(x)
        x = torch.sigmoid(x)
        x = x.view(-1, 1)
        return x


if __name__ == "__main__":
    #Generatror
    latent_vector = torch.randn([10,100], dtype=torch.float)
    G_net = G()
    Gen_img = G_net(latent_vector)
    print(Gen_img.shape)
    
    #Discriminator
    x = torch.randn([10,1,28,28], dtype=torch.float)
    D_net = D()
    y = D_net(x)
    print(y.shape)