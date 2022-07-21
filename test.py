import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST
import torchvision.utils as vutils

import matplotlib.pyplot as plt
import numpy as np

from model import Generator, Discriminator

# Device
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

# Model
z_dim = 100
batch_size = 32
G = Generator(z_dim, 64, 1).to(device)
G.load_state_dict(torch.load('generator'))

# MNIST Dataset
download_root = './MNIST_datset/'
fashion_root = './fashionMNIST_datset/'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=1.0),
    transforms.Resize(64, interpolation=transforms.InterpolationMode.BICUBIC)
])

train_dataset = MNIST(download_root, train=True, transform=transform, download=True)
test_dataset = MNIST(download_root, train=False, transform=transform, download=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

fashion_dataset = FashionMNIST(fashion_root, train=False, transform=transform, download=True)
fashion_dataloader = DataLoader(dataset=fashion_dataset, batch_size=batch_size, shuffle=False)

# Generate fake images
z = torch.randn(batch_size, z_dim, 1, 1, device=device)
fake_images = G(z)

# Real image Visualization
real_batch = next(iter(train_dataloader))
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device), padding=5, normalize=True).cpu(), (1,2,0)))

# Fake image Visualization
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(vutils.make_grid(fake_images.to(device), padding=5, normalize=True).cpu(), (1,2,0)))
# plt.savefig('Fakeimages.png')
plt.show()