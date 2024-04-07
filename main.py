# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 20:18:49 2024

@author: Tuan
"""

import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from model import ADJSCC_V
from data_loader import ImgData

# import starGAN_model as starGAN
# import cycleGAN_model as cycleGAN
# from CNNclassifier_model import DigitClassifierCNN

from utils import EarlyStopping

DS_NAME = ["MNIST", "MNISTM", "SYN", "USPS"]

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4

CHANNEL = 'AWGN'  # Choose AWGN or Fading
N_CHANNELS = 256
KERNEL_SIZE = 5

IMAGE_SIZE = 32

SAVE_EVERY = 5

enc_out_shape = [48, IMAGE_SIZE//4, IMAGE_SIZE//4]

for ds_idx in range(len(DS_NAME)):

    current_setting = f'{DS_NAME[ds_idx]}_batch{BATCH_SIZE}_{CHANNEL}_noChannels{N_CHANNELS}_kernelsz{KERNEL_SIZE}'
    print(f'Current setting: {current_setting}')

    model_path = os.path.join('.', 'model', current_setting)
    os.makedirs(model_path, exist_ok=True)

    train_ds = ImgData(f'./dataset/{DS_NAME[ds_idx]}_train.pt', IMAGE_SIZE, IMAGE_SIZE)
    test_ds = ImgData(f'./dataset/{DS_NAME[ds_idx]}_test.pt', IMAGE_SIZE, IMAGE_SIZE)


    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_ds,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True)

    DeepJSCC_V = ADJSCC_V(enc_out_shape, KERNEL_SIZE, N_CHANNELS).cuda()
    # DeepJSCC_V = nn.DataParallel(DeepJSCC_V)

    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(DeepJSCC_V.parameters(), lr=LEARNING_RATE)

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(
    #     optimizer,
    #     gamma=0.9,
    #     verbose=True
    # )

    bestLoss = np.Inf

    # early_stopping = EarlyStopping(model_save_path, patience=10, verbose=True, criteria='eval loss')

    train_info = []
    eval_info = []

    for epoch in range(EPOCHS):

        DeepJSCC_V.train()

        # Model training
        train_loss = 0.0

        for batch_idx, (x_input, _) in tqdm(enumerate(train_loader),
                                            total=len(train_loader),
                                            desc=f'Epoch {epoch+1} training progress'):
            # print(batch_idx)%
            x_input = x_input.cuda()

            SNR_TRAIN = torch.randint(0, 28, (x_input.shape[0], 1)).cuda()
            CR = 0.1+0.9*torch.rand(x_input.shape[0], 1).cuda()

            x_rec =  DeepJSCC_V(x_input, SNR_TRAIN, CR, CHANNEL)

            loss = criterion(x_input, x_rec)
            # loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_input.size(0)

        train_loss /= len(train_ds)

        print()
        print(f'Average train loss: {train_loss}')
        print("------------------------------")

        train_info.append([train_loss])

        # Learning rate scheduling
        # scheduler.step()

        # Model Evaluation

        DeepJSCC_V.eval()
        eval_loss = 0

        with torch.no_grad():

            for batch_idx, (test_input, _) in enumerate(test_loader):

                test_input = test_input.cuda()

                SNR_TEST = torch.randint(0, 28, (test_input.shape[0], 1)).cuda()
                CR = 0.1+0.9*torch.rand(test_input.shape[0], 1).cuda()

                test_rec =  DeepJSCC_V(test_input, SNR_TEST, CR, CHANNEL)

                eval_loss += criterion(test_input, test_rec).item() * test_input.size(0)

            eval_loss = eval_loss / len(test_ds)

        print(f'Average eval loss: {eval_loss}')
        print("==============================")

        eval_info.append([eval_loss])

        # early_stopping(eval_loss, DeepJSCC_V)
        # if early_stopping.early_stop:
        #     print(f"Early stopping triggered at epoch {epoch}.\n")
        #     break

        if (epoch+1) % SAVE_EVERY == 0:
            model_name = f'JSCC-V_{DS_NAME[ds_idx]}_epoch_{epoch+1}.pt'
            model_save_path = os.path.join(model_path, model_name)
            torch.save(DeepJSCC_V.state_dict(), model_save_path)

    df = pd.DataFrame(train_info, columns=['Train Loss'])
    train_info_path = os.path.join('.', 'train_info', current_setting)
    os.makedirs(train_info_path, exist_ok=True)
    train_info_path = os.path.join(train_info_path, 'train_info.csv')
    df.to_csv(train_info_path, index=True)

    print(f'Saving train info to {train_info_path}')

    df = pd.DataFrame(eval_info, columns=['Eval Loss'])
    eval_info_path = os.path.join('.', 'eval_info', current_setting)
    os.makedirs(eval_info_path, exist_ok=True)
    eval_info_path = os.path.join(eval_info_path, 'eval_info.csv')
    df.to_csv(eval_info_path, index=True)

    print(f'Saving eval info to {eval_info_path}')
    print()

print('All done!')































