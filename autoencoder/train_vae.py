import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchvision.utils import save_image
from tqdm import tqdm

plt.style.use('ggplot')

from autoencoder.models.variational_autoencoder import VAE
from autoencoder.dataloader.dataloader import DataLoader
from torch.nn import functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("-ae", "--ae-path", help="Path to saved autoencoder (otherwise start from scratch)", type=str)
parser.add_argument(
    "-f", "--folders", help="Path to folders containing images for training", type=str, nargs="+", required=True
)
parser.add_argument("--z-size", help="Latent space", type=int, default=64)
parser.add_argument("--seed", help="Random generator seed", type=int)
parser.add_argument("--n-samples", help="Max number of samples", type=int, default=-1)
parser.add_argument("-bs", "--batch-size", help="Batch size", type=int, default=64)
parser.add_argument("--learning-rate", help="Learning rate", type=float, default=1e-4)
parser.add_argument("--n-epochs", help="Number of epochs", type=int, default=25)
parser.add_argument("--verbose", help="Verbosity", type=int, default=1)
parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
args = parser.parse_args()


class AddGaussianNoise(object):
    """
    Add noise to the input image to improve learning robustness
    see: https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745/2
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def save_reconstructed_images(recon_images, epoch):
    save_image(recon_images.cpu(), f"training-images/track/reconstructed{epoch}.jpg")


def save_noisy_images(recon_images, epoch):
    save_image(recon_images.cpu(), f"training-images/track/noisy{epoch}.jpg")


def save_raw_images(recon_images, epoch):
    save_image(recon_images.cpu(), f"training-images/track/raw{epoch}.jpg")


def add_noise(inputs, noise_factor=0.3):
    gaussian = AddGaussianNoise(0.05, 0.05)
    inputs = gaussian(inputs)
    noisy = inputs + torch.randn_like(inputs) * noise_factor
    noisy = torch.clip(noisy, 0., 1.)
    return noisy


def train(model, dataloader, dataset_size, device, criterion):
    model.train()
    running_loss = 0.0
    counter = 0
    noise_factor = 0.3
    train_loss = 0
    for i, data_from_loader in tqdm(enumerate(dataloader), total=int(dataset_size / dataloader.batch_size)):
        counter += 1
        image_tensor = data_from_loader[0]
        noisy_image = add_noise(image_tensor, noise_factor)

        obs = noisy_image.to(device)
        raw_obs = image_tensor.to(device)
        model.optimizer.zero_grad()
        predicted_obs, mu, logvar = model.forward(obs)
        bce_loss = criterion(predicted_obs, raw_obs)
        loss = model.final_loss(bce_loss, mu, logvar)

        loss.backward()
        train_loss += loss.item()
        model.optimizer.step()
    return train_loss


def validate(model, dataloader, dataset, device, criterion):
    model.eval()
    running_loss = 0.0
    counter = 0
    val_loss = 0
    val_loss1 = 0
    noise_factor = 0.3
    with torch.no_grad():
        for i, data_from_loader in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
            counter += 1

            image_tensor = data_from_loader[0]
            noisy_image = add_noise(image_tensor, noise_factor)
            obs = noisy_image.to(device)
            raw_obs = image_tensor.to(device)
            predicted_obs, mu, logvar = model.forward(obs)
            bce_loss = criterion(predicted_obs, raw_obs)
            loss = model.final_loss(bce_loss, mu, logvar)
            val_loss += loss.item()

            # save the last full batch input and output of every epoch
            if i == int(len(dataset) / dataloader.batch_size) - 2:
                sample_predicted_images = predicted_obs
                sample_noisy_image = noisy_image
                sample_raw_obs = raw_obs

    return val_loss, sample_predicted_images, sample_noisy_image, sample_raw_obs


###########################################################################
# remove this later
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
    save_image(recon_images.cpu(), f"training-images/track/vae_single{epoch+1}.jpg")


def pull_image():
    image_idx = np.random.randint(n_samples)
    image = Image.open(images[image_idx])
    return image


def pull_and_convert_image():
    image_idx = np.random.randint(n_samples)
    # image = cv2.imread(images[image_idx])
    image = Image.open(images[image_idx])
    # image.show()
    transform = transforms.Compose([
        transforms.Resize((80, 160)),
        transforms.ToTensor(),
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    print(image.size())
    return image


####################################################################

# pull_and_convert_image()

model = VAE(z_size=args.z_size, c_hid=64, num_image_channels=3, learning_rate=args.learning_rate).to(device)
# model = Autoencoder(image_channels=3,
#                     base_channel_size=32,
#                     latent_dim=32,
#                     flatten_size=6144,
#                     learning_rate=0.0001).to(device)
print(model)

dataloader = DataLoader("images/track", 64)
ae_id = int(time.time())
writer = SummaryWriter(log_dir=f'runs/track/vae/{ae_id}')

# set the learning parameters
lr = 0.0003
epochs = 200
batch_size = 8
best_loss = np.inf
save_path = "logs/track/vae-{}_{}.pkl".format(args.z_size, ae_id)
best_model_path = "logs/track/vae-{}_{}_best.pkl".format(args.z_size, ae_id)
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# we're using Adam optimizer here again
optimizer = optim.Adam(model.parameters(), lr=lr)

# Using Binary Cross Entropy loss
criterion = nn.BCELoss(reduction='sum')

train_loss = []
valid_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch + 1} of {epochs}")

    train_epoch_loss = train(
        model, dataloader.train_dataloader(), len(dataloader.train_dataset()), device, criterion
    )
    valid_epoch_loss, recon_images, noisy_images, raw_images = validate(
        model, dataloader.val_dataloader(), dataloader.val_dataset(), device, criterion
    )

    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)

    # save the reconstructed images from the validation loop
    save_reconstructed_images(recon_images, epoch + 1)
    # save_noisy_images(noisy_images, epoch + 1)
    # save_raw_images(raw_images, epoch + 1)
    # convert the reconstructed images to PyTorch image grid format
    recon_image_grid = make_grid(recon_images.detach().cpu())
    noisy_image_grid = make_grid(noisy_images.detach().cpu())
    raw_image_grid = make_grid(raw_images.detach().cpu())

    writer.add_image('raw_images', raw_image_grid, epoch)
    writer.add_image('noisy_images', noisy_image_grid, epoch)
    writer.add_image('reconstructed_images', recon_image_grid, epoch)
    writer.add_scalar('loss/train', np.mean(train_loss), epoch)
    writer.add_scalar('loss/val', np.mean(valid_loss), epoch)

    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {valid_epoch_loss:.4f}")

    if valid_epoch_loss < best_loss:
        best_loss = valid_epoch_loss
        print("Saving best model to {}".format(best_model_path))
        model.save(best_model_path)

    ##################################
    ## remove this

    irweazle = pull_image()
    encoded_image = model.encode_raw_image(irweazle)
    something = model.decode_forward(encoded_image)
    something = something.to(device)
    save_single_reconstructed_images(something, epoch)
    print('something')

writer.flush()
writer.close()
