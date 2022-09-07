import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils

a = torch.Tensor(1, 100, 1, 1)
a = a.numpy()
a = np.array([a])
print(a.shape)
