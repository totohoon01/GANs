# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 18:00:22 2021

@author: user
"""
import torch
from torch import nn, optim
import numpy as np
from net import G, D
from data_processing import load_data
import cv2
import os
import tqdm

if not os.path.isdir("./gen_images"):
    os.mkdir("./gen_images")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#NETOWORKS
Generator = G()
Generator = Generator.to(device)
Discriminator = D()
Discriminator = Discriminator.to(device)

criterion = nn.BCELoss()
G_optimizer = optim.Adam(Generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
D_optimizer = optim.Adam(Discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
batch_size = 1000

#FIXED Z for generating image
fixed_z = torch.randn([100], dtype=torch.float).to(device)

target_real = torch.ones(batch_size, 1).to(device)
target_fake = torch.zeros(batch_size, 1).to(device)

train_data = load_data(train=True, batch_size=batch_size)
for epoch in range(100):
    for X, _ in tqdm.tqdm(train_data):
        X = X.to(device)
        
        #Discriminate Real_image
        D_pred_real = Discriminator(X)
        D_loss_real = criterion(D_pred_real, target_real) #real -> real
        
        #Random vector for Generating Fake Image
        z = torch.randn([batch_size, 100], dtype=torch.float)
        z = z.to(device)
        
        #Discriminate Fake_image
        fake_image = Generator(z)
        D_pred_fake = Discriminator(fake_image)
        D_loss_fake = criterion(D_pred_fake, target_fake) #fake -> fake
        
        #TRAIN Discriminator
        D_loss = D_loss_real + D_loss_fake
        Discriminator.zero_grad()
        D_loss.backward()
        D_optimizer.step()
        
        
        #Generate Fake image for trining Generator
        fake_image = Generator(z)
        D_pred_fake = Discriminator(fake_image)
        
        #Train Generator
        G_loss = criterion(D_pred_fake, target_real) #fake -> real
        Generator.zero_grad()
        G_loss.backward()
        G_optimizer.step()
        
    print(f"\t epoch:{epoch}, D_loss:{D_loss}, G_loss:{G_loss}")
    #Testing Generator!!
    gen_img = Generator(fixed_z)
    gen_img = gen_img.view(28,28)
    gen_img = gen_img.cpu().detach().numpy()
    cv2.imwrite(f"./gen_images/epoch_{epoch}.png", gen_img)
    