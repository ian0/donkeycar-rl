import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torch import nn, optim
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import os
import random
import time
import cv2
from PIL import Image
from matplotlib import cm

plt.style.use('ggplot')

from models.autoencoder import Autoencoder
from dataloader.dataloader import DataLoader
from torch.nn import functional as F
from configs.config import ROI, INPUT_DIM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("-ae", "--ae-path", help="Path to saved autoencoder (otherwise start from scratch)",
                    type=str, default='logs/ae-32_1627814771_best.pkl')
parser.add_argument(
    "-f", "--folders", help="Path to folders containing images for training", type=str, nargs="+",
    required=True
)
parser.add_argument("--z-size", help="Latent space", type=int, default=32)
parser.add_argument("--seed", help="Random generator seed", type=int)
parser.add_argument("--n-samples", help="Max number of samples", type=int, default=-1)
parser.add_argument("-bs", "--batch-size", help="Batch size", type=int, default=64)
parser.add_argument("--learning-rate", help="Learning rate", type=float, default=1e-4)
parser.add_argument("--n-epochs", help="Number of epochs", type=int, default=200)
parser.add_argument("--verbose", help="Verbosity", type=int, default=1)
parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
args = parser.parse_args()

#model = Autoencoder(32)
model = Autoencoder.load(args.ae_path)
#args.z_size = model.z_size
model.eval()

folders, images = [], []
for folder in args.folders:
    if not folder.endswith("/"):
        folder += "/"
    folders.append(folder)
    images_ = [folder + im for im in os.listdir(folder) if im.endswith(".jpg")]
    print("{}: {} images".format(folder, len(images_)))
    images.append(images_)


images = np.concatenate(images)
n_samples = len(images)

def show_img(img):
    plt.figure()
    img = img.permute(1, 2, 0)
    plt.imshow(img.numpy())
    plt.show()


def save_single_reconstructed_images(recon_images, epoch):
    save_image(recon_images.cpu(), f"training-images/trained_model{epoch}.jpg")



def pull_and_convert_image():
    image_idx = np.random.randint(n_samples)
    # image = cv2.imread(images[image_idx])
    image = Image.open(images[image_idx])
    #image.show()
    transform = transforms.Compose([
        transforms.Resize((80, 160)),
        transforms.ToTensor(),
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    print(image.size())
    return image


irweazle = pull_and_convert_image()
irweazle = irweazle.to(device)
encoded_image = model.encode_forward(irweazle)
something = model.decode_forward(encoded_image)
#something = model.forward(some_image)
something = something.to(device)
save_single_reconstructed_images(something, 1)
print('something')
