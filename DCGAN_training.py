import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
import torchvision.transforms as transforms
import torchvision.utils as vutils

import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time

from model import Generator, Discriminator

# Device
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

# Random Seed
seed = random.randint(1, 10000)
torch.manual_seed(seed)

# Training parameters
parser = argparse.ArgumentParser(description="PyTorch SR")
parser.add_argument("--batchSize", type=int, default=10, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=50, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=200, help="Learning Rate Scheduler Step")
parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
parser.add_argument("--weight-decay", default=1e-4, type=float, help="weight decay, Default: 0")
opt = parser.parse_args()

# MNIST Dataset
download_root = './MNIST_datset/'
fashion_root = './fashionMNIST_datset/'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=1.0),
    transforms.Resize(64, interpolation=transforms.InterpolationMode.BICUBIC)
])

train_dataset = MNIST(download_root, train=True, transform=transform, download=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batchSize, shuffle=True)

# Test Image
index = 1

# def image_show(img):
#     img = torch.squeeze(img)
#     img = img.numpy()
#     plt.imshow(img, 'gray')
#     plt.show()
#
# for batch, (inputs, targets) in enumerate(train_dataloader):
#     image_show(inputs[index])
#     break
#
# for batch, (inputs, targets) in enumerate(fashion_dataloader):
#     image_show(inputs[index])
#     break

# Model
z_dim = 100
G = Generator(z_dim, 64, 1).to(device)
D = Discriminator(z_dim, 1).to(device)

# Optimizer
G_optim = optim.Adam(G.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, betas=(0.9, 0.99), eps=1e-08)
D_optim = optim.Adam(D.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, betas=(0.9, 0.99), eps=1e-08)
criterion = nn.BCELoss().to(device)

# Training Adversarial Network
img_list = []
G_losses = []
D_losses = []

def train():
    iters = 0

    for i, (images, values) in enumerate(train_dataloader):
        real_images = images.to(device)
        batch_size = real_images.size(0)
        z = torch.randn(batch_size, z_dim, 1, 1, device=device)

        # Discriminator: maximize log(D(x)) + log(1-D(G(z)))
        D.zero_grad()

        real_output = D(real_images)[1]
        real_label = torch.ones_like(real_output, device=device)
        fake_images = G(z)
        fake_output = D(fake_images.detach())[1]
        fake_label = torch.zeros_like(fake_output, device=device)

        errorD_real = criterion(real_output, real_label)
        errorD_fake = criterion(fake_output, fake_label)
        errorD = errorD_real + errorD_fake
        errorD.backward()
        D_optim.step()

        P_real = real_output.mean().item()
        P_fake = fake_output.mean().item()  # discriminator가 진짜라고 판별할 확률

        # Generator: maximize log(D(G(z)))
        G.zero_grad()

        fake_images = G(z)
        fake_output = D(fake_images)[1]
        fake_label = torch.ones_like(fake_output, device=device)
        errorG = criterion(fake_output, fake_label)
        errorG.backward()
        G_optim.step()

        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
              % (epoch, opt.nEpochs, i + 1, len(train_dataloader),
                 errorD.item(), errorG.item()))

        G_losses.append(errorG.item())
        D_losses.append(errorD.item())

        # Show generator output and check whether it is working right
        if iters % 500 == 0 or (i == len(train_dataloader) - 1):
            with torch.no_grad():
                fake = G(z).detach().cpu()
            img_list.append(vutils.make_grid(fake, normalize=True))

        iters += 1


start = time.time()
for epoch in range(1, opt.nEpochs+1):
    train()

print("time: ", time.time()-start)

# Save Model
torch.save(G.state_dict(), 'generator')
torch.save(D.state_dict(), 'discriminator')

# Loss Visualization
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('GDloss.png')

# Real image Visualization
real_batch = next(iter(train_dataloader))
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:10], padding=5, normalize=True).cpu(), (1,2,0)))

# Fake image Visualization
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.savefig('RealFakeimages.png')
plt.show()