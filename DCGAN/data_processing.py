# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 18:00:36 2021

@author: user
"""
import os
import torch
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset, DataLoader


path = "./dataset"
if not os.path.isdir(path):
    os.mkdir(path)

#왜인지 학교 랜을 이용하면 데이터가 다운로드가 안된다.
train_data = MNIST(path, train=True, download=True)
test_data = MNIST(path, train=False, download=True)


def load_data(train=True, batch_size=100):
    name = "training.pt" if train else "test.pt"
    pt_file = torch.load('./dataset/MNIST/processed/' + name)
    data, label = pt_file
    data = data.view(-1,1,28,28).float()
    label = label.view(-1, 1)
    ds = TensorDataset(data, label)
    data_loader = DataLoader(ds, batch_size=batch_size)
    return data_loader

if __name__ == "__main__":
    data_loader = load_data(0)