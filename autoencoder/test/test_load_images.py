import random

# from vae import VAE
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils

from src.vae.dataloader.dataloader import DataLoader
from src.vae.model import CNN_VAE

'''
Code to display grid taken from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
'''


def display_origional_image(device, dataloader):
    with torch.no_grad():
        for data in random.sample(list(dataloader()), 1):
            imgs, _ = data
            imgs = imgs.to(device)
            img = np.transpose(imgs[0].cpu().numpy(), [1, 2, 0])
            plt.subplot(121)
            plt.imshow(np.squeeze(img))
            plt.show()
            break


def display_origional_and_random_image(device, dataloader):
    model_path = "../cnn_vae-32-dict.pt"
    vae_model = CNN_VAE().to(device)
    vae_model.load_state_dict(torch.load(model_path))
    vae_model.to(device)
    vae_model.eval()

    with torch.no_grad():
        for data in random.sample(list(dataloader), 1):
            imgs, _ = data
            imgs = imgs.to(device)
            img = np.transpose(imgs[0].cpu().numpy(), [1, 2, 0])
            plt.subplot(121)
            plt.imshow(np.squeeze(img))
            out, mu, logVAR = vae_model(imgs)
            outimg = np.transpose(out[0].cpu().numpy(), [1, 2, 0])
            plt.subplot(122)
            plt.imshow(np.squeeze(outimg))
            plt.show()
            break

def display_origional_and_random_image_take2(device, dataloader):
    model_path = "../cnn_vae-32-dict.pt"
    vae_model = CNN_VAE().to(device)
    vae_model.load_state_dict(torch.load(model_path))
    vae_model.to(device)
    vae_model.eval()

    with torch.no_grad():
        for data in random.sample(list(dataloader), 1):
            imgs, _ = data
            imgs = imgs.to(device)
            img = np.transpose(imgs[0].cpu().numpy(), [1, 2, 0])
            plt.subplot(121)
            plt.imshow(np.squeeze(img))
            out, mu, logVAR = vae_model(imgs)
            outimg = np.transpose(out[0].cpu().numpy(), [1, 2, 0])
            plt.subplot(122)
            plt.imshow(np.squeeze(outimg))
            plt.show()
            break


def display_origional_image_grid(device, dataloader):
    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()


def display_reconstructed_image_grid(device, dataloader):
    model_path = "../cnn_vae-32-dict.pt"
    net = CNN_VAE().to(device)
    net.load_state_dict(torch.load('cnn_vae-32-dict.pt'))
    net.to(device)
    net.eval()

    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dl = DataLoader('../images', 1)
    loader = dl.val_dataloader()
    # display_random_image(device, loader)
    # display_origional_image_grid(device, loader)
    display_origional_and_random_image(device, loader)
