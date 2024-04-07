# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 21:25:46 2024

@author: Tuan
"""

import torch

class EarlyStopping:
    def __init__(self, model_name, patience=10, sign=-1, verbose=False, criteria='loss', epsilon=0.0):
        """
        Args:
            patience (int): How long to wait after last time improved.
                            Default: 10
            verbose (bool): If True, prints a message for each improvement.
                            Default: False
            epsilon (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0.0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.epsilon = epsilon
        self.sign = sign
        self.save_path = model_name

        if self.verbose:
            self.sign_str = '>' if sign < 0 else '<'
            self.sign_str2 = 'decreased' if sign < 0 else 'increased'
            self.criteria = criteria

    def __call__(self, score, model):

        score = score * self.sign

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)

        elif score < self.best_score * (1 + self.epsilon * self.sign):
            self.counter += 1

            if self.verbose:
                print(f'\nCurrent {self.criteria} {score * self.sign} {self.sign_str} {self.best_score * self.sign} * {1 + self.epsilon * self.sign} = {self.best_score * self.sign * (1 + self.epsilon * self.sign)}')
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(score, model)
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, score, model):

        if self.verbose:
            print(f'\n{self.criteria} {self.sign_str2} ({self.best_score * self.sign} --> {score * self.sign}).  Saving model ...\n')
            print(f'Saving model to {self.save_path}')
        # Note: Here you should define how you want to save your model. For example:
        torch.save(model.state_dict(), self.save_path)