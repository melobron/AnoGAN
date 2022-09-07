import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from model import Generator, Discriminator

# Device
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

# Parameters
batch_size = 1
lr = 1e-4
threshold = 50

# Model
z_dim = 100
G = Generator(z_dim, 64, 1).to(device)
D = Discriminator(z_dim, 1).to(device)
G.load_state_dict(torch.load('generator'))
D.load_state_dict(torch.load('discriminator'))

# MNIST Dataset
download_root = './MNIST_datset/'
fashion_root = './fashionMNIST_datset/'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=1.0),
    transforms.Resize(64, interpolation=transforms.InterpolationMode.BICUBIC)
])

test_dataset = MNIST(download_root, train=False, transform=transform, download=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

fashion_dataset = FashionMNIST(fashion_root, train=False, transform=transform, download=True)
fashion_dataloader = DataLoader(dataset=fashion_dataset, batch_size=batch_size, shuffle=False)

# Loss
def residual_loss(real, generated):
    difference = real - generated
    return torch.sum(torch.abs(difference))

def discrimination_loss(discriminator, real, generated):
    real_feature = discriminator(real)[0]
    generated_feature = discriminator(generated)[0]
    difference = real_feature - generated_feature
    return torch.sum(torch.abs(difference))

# Anomaly Score
def total_loss(r_loss, d_loss, parameter=0.1):
    return (1 - parameter) * r_loss + parameter * d_loss

def anomaly_score(discriminator, real, generated):
    r_loss = residual_loss(real, generated)
    d_loss = discrimination_loss(discriminator, real, generated)
    a_loss = total_loss(r_loss, d_loss, parameter=0.5)
    return a_loss.cpu().data.numpy()

# Z Update
G.eval()
D.eval()

z = torch.randn(1, z_dim, 1, 1, device=device, requires_grad=True)
z_optim = optim.Adam([z], lr=lr)

latent_space = []

for index, (images, values) in enumerate(fashion_dataloader):
    real_images = images.to(device)
    for step in range(30001):
        generated_images = G(z)
        z_optim.zero_grad()
        r_loss = residual_loss(real_images, generated_images)
        d_loss = discrimination_loss(D, real_images, generated_images)
        a_loss = total_loss(r_loss, d_loss, parameter=0.1)
        a_loss.backward(retain_graph=True)
        z_optim.step()

        if step % 1000 == 0:
            loss = a_loss.item()  # == cpu().numpy()
            print("[step: %d] \t Total loss: %.4f" %(step, loss))

        if step == 30000:
            latent_space.append(z.cpu().data.numpy())
    break

# Evaluation
for index, (images, values) in enumerate(fashion_dataloader):
    real_images = images.to(device)
    print(real_images.shape)
    latent_space = np.array(latent_space[0])
    latent_space = torch.Tensor(latent_space).to(device)
    generated_images = G(latent_space).to(device)

    score = anomaly_score(D, real_images, generated_images)
    score = np.round(score, 2)

    real_images = np.transpose(real_images.cpu().data.numpy().squeeze(0), axes=(1, 2, 0)) * 255
    generated_images = np.transpose(generated_images.cpu().data.numpy().squeeze(0), axes=(1, 2, 0)) * 255

    diff_image = real_images - generated_images
    diff_image[diff_image <= threshold] = 0

    fig, plots = plt.subplots(1, 3)
    fig.suptitle(f'Anomaly - (anomaly score: {score:.4f}) \n Class {values.item()}')

    fig.set_figwidth(9)
    fig.set_tight_layout(True)
    plots = plots.reshape(-1)
    plots[0].imshow(real_images.astype('uint8'), cmap='gray', label='real')
    plots[1].imshow(generated_images.astype('uint8'), cmap='gray')
    plots[2].imshow(diff_image.astype('uint8'), cmap='gray')

    plots[0].set_title('real')
    plots[1].set_title('generated')
    plots[2].set_title('difference')
    plt.show()

    break