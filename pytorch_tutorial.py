# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/pytorch_basics/main.py
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.Tensor(1., requires_grad=True)
w = torch.Tensor(2., requires_grad=True)
b = torch.Tensor(3., requires_grad=True)