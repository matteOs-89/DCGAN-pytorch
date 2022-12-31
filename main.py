import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

import random
import time
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

######################
# Data Preprocessing #
######################

dir = "/content/drive/MyDrive/afhq/train"

image_size = 64
batch_size = 64
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

train_data = ImageFolder(dir, transform=T.Compose([
    T.Resize(image_size),
    T.CenterCrop(image_size),
    T.ToTensor(),
    T.Normalize(*stats),
    T.RandomHorizontalFlip(p=0.5)
]))

train_dl = DataLoader(train_data, batch_size, shuffle=True, num_workers=2, drop_last=True)

x, y = next(iter(train_dl))
print(f"X shape: {x.shape}")
print(f"y shape: {y.shape}")
x[0]

sample_dir = "resultsGAN"

if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]


def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))


def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break


show_batch(train_dl)


class discriminatorNet(nn.Module):

    def __init__(self):
        super().__init__()

        # convolution layers
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.conv5 = nn.Conv2d(512, 1, 4, 2, 0, bias=False)

        # batchnorm
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.bn2(self.conv2(x)), .2)

        x = F.leaky_relu(self.bn3(self.conv3(x)), .2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), .2)

        return torch.sigmoid(self.conv5(x)).view(-1, 1)


Dnet = discriminatorNet()

y = Dnet(torch.randn(10, 3, 512, 512))

y.shape


class generatorNet(nn.Module):

    def __init__(self):
        super().__init__()

        # convolution layers
        self.conv1 = nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False)
        self.conv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        self.conv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.conv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.conv5 = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)

        # batchnorm
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.relu(self.bn2(self.conv1(x)))

        x = F.relu(self.bn3(self.conv2(x)))
        x = F.relu(self.bn4(self.conv3(x)))

        x = F.relu(self.bn5(self.conv4(x)))
        x = torch.tanh(self.conv5(x))

        return x


Gnet = generatorNet()

y = Gnet(torch.randn(10, 100, 1, 1))
print(y.shape)

loss_fun = nn.BCELoss()

Dnet = discriminatorNet()
Gnet = generatorNet()

D_optim = torch.optim.Adam(Dnet.parameters(), lr=0.0003, betas=(0.5, .999))
G_optim = torch.optim.Adam(Gnet.parameters(), lr=0.0003, betas=(.5, .999))

D_loss = []
G_loss = []
losses = []
disDecs = []

num_epoch = 500

for epoch in range(num_epoch):

    starttime = time.time()

    for image, _ in train_dl:
        Img_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        """Train discriminator"""

        img_pred = Dnet(image[:64])
        d_loss_real = loss_fun(img_pred, Img_labels)

        fake_image = Gnet(torch.randn(batch_size, 100, 1, 1))
        fake_pred = Dnet(fake_image)
        d_loss_fake = loss_fun(fake_pred, fake_labels)

        d_loss = d_loss_fake + d_loss_real
        D_loss.append(d_loss.detach())

        # back prop
        D_optim.zero_grad()
        d_loss.backward()
        D_optim.step()

        """Train Generator"""

        fake_images = Gnet(torch.randn(batch_size, 100, 1, 1))
        pred_fake = Dnet(fake_images)

        g_loss = loss_fun(pred_fake, Img_labels)
        G_loss.append(g_loss.detach())

        G_optim.zero_grad()
        g_loss.backward()
        G_optim.step()

        losses.append([d_loss.item(), g_loss.item()])

        timekeeping = time.time() - starttime

        D1 = torch.mean((img_pred > .5).float()).detach()
        D2 = torch.mean((pred_fake > .5).float()).detach()
        disDecs.append([D1, D2])

    """Save images after epochs and training updates"""
    if (epoch + 1) % 3 == 0:
        save_image(fake_images.view(-1, 3, 64, 64), "resultsGAN/sample_Train" + str(epoch) + ".jpg")

    if (epoch + 1) % 3 == 0:
        msg = f"Finshed epoch {epoch + 1}/{num_epoch}..time taken: {timekeeping}"
        sys.stdout.write("\r" + msg)

"""Evaluating Generator performance after training"""

with torch.no_grad():
    Gnet.eval()
    fake_data = Gnet(torch.randn(batch_size, 100, 1, 1))
    save_image(fake_data.view(-1, 3, 64, 64), "resultsGAN/sample_" + str(epoch) + ".jpg")

    # Visualization
    fig, ax = plt.subplots(3, 6, figsize=(8, 6))

    for i, index in enumerate(ax.flatten()):
        index.imshow(fake_data[i,].detach().squeeze().permute(1, 2, 0), cmap="jet")
        index.axis("off")

plt.show()

"""Training loss visualization"""

plt.plot(D_loss, "--o")
plt.plot(G_loss, "--o")
plt.xlabel("steps")
plt.legend()
plt.show()