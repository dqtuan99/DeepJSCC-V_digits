# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:49:04 2024

@author: Tuan
"""

import torch
from torchvision import transforms
import numpy as np


MNIST_train_path = './dataset/MNIST_train.pt'
MNISTM_train_path = './dataset/MNISTM_train.pt'
SYN_train_path = './dataset/SYN_train.pt'
USPS_train_path = './dataset/USPS_train.pt'

MNIST_test_path = './dataset/MNIST_test.pt'
MNISTM_test_path = './dataset/MNISTM_test.pt'
SYN_test_path = './dataset/SYN_test.pt'
USPS_test_path = './dataset/USPS_test.pt'


class ImgData(torch.utils.data.Dataset):
    def __init__(self, path, w, h):

        self.transform = transforms.Compose([transforms.Resize([w, h]),
                                             transforms.Normalize([0.5], [0.5])])

        self.data = torch.load(path)

        self.img = self.transform(self.data[0])
        self.img = self.pre_processing(self.img)

        self.label = self.data[1]

        self.len = self.label.shape[0]


    def pre_processing(self, img):
        if len(img.shape) < 4:
            img = img.unsqueeze(1).repeat(1, 3, 1, 1)

        return img

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        return self.img[index], self.label[index]


class AllDomainData(torch.utils.data.Dataset):

    def __init__(self, w, h, is_training=True, DA_task=False):

        self.transform = transforms.Compose([transforms.Resize([w, h]),
                                            transforms.Normalize([0.5], [0.5])])

        self.DA_task = DA_task

        if is_training:
            self.MNIST_data = torch.load(MNIST_train_path)
            self.MNISTM_data = torch.load(MNISTM_train_path)
            self.SYN_data = torch.load(SYN_train_path)
            self.USPS_data = torch.load(USPS_train_path)

        else:
            self.MNIST_data = torch.load(MNIST_test_path)
            self.MNISTM_data = torch.load(MNISTM_test_path)
            self.SYN_data = torch.load(SYN_test_path)
            self.USPS_data = torch.load(USPS_test_path)

        self.MNIST_img = self.transform(self.MNIST_data[0])
        self.MNISTM_img = self.transform(self.MNISTM_data[0])
        self.SYN_img = self.transform(self.SYN_data[0])
        self.USPS_img = self.transform(self.USPS_data[0])

        self.MNIST_label = self.MNIST_data[1]
        self.MNISTM_label = self.MNISTM_data[1]
        self.SYN_label = self.SYN_data[1]
        self.USPS_label = self.USPS_data[1]

        if DA_task:
            self.MNIST_img, self.MNIST_domain = self.pre_processing(self.MNIST_img, 0)
            self.MNISTM_img, self.MNISTM_domain = self.pre_processing(self.MNISTM_img, 1)
            self.SYN_img, self.SYN_domain = self.pre_processing(self.SYN_img, 2)
            self.USPS_img, self.USPS_domain = self.pre_processing(self.USPS_img, 3)
        else:
            self.MNIST_img = self.pre_processing(self.MNIST_img)
            self.MNISTM_img = self.pre_processing(self.MNISTM_img)
            self.SYN_img = self.pre_processing(self.SYN_img)
            self.USPS_img = self.pre_processing(self.USPS_img)

        self.img = torch.vstack((self.MNIST_img,
                                 self.MNISTM_img,
                                 self.SYN_img,
                                 self.USPS_img))

        self.label = torch.hstack((self.MNIST_label,
                                   self.MNISTM_label,
                                   self.SYN_label,
                                   self.USPS_label))

        if DA_task:
            self.domain = np.hstack((self.MNIST_domain,
                                     self.MNISTM_domain,
                                     self.SYN_domain,
                                     self.USPS_domain))

        self.len = self.label.shape[0]


    def pre_processing(self, img, domain=None):

        if len(img.shape) < 4:
            img = img.unsqueeze(1).repeat(1, 3, 1, 1)

        if domain is not None:
            num_img = img.shape[0]
            domain_label = np.zeros(num_img, dtype=int) + domain

            return img, domain_label

        return img

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        if self.DA_task:
            return self.img[index], self.label[index], self.domain[index]

        return self.img[index], self.label[index]