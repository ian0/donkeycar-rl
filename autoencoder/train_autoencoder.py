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

from autoencoder.models.autoencoder import Autoencoder, Print
from autoencoder.dataloader.dataloader import DataLoader
from torch.nn import functional as F
from configs.config import ROI, INPUT_DIM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("-ae", "--ae-path", help="Path to saved autoencoder (otherwise start from scratch)", type=str)
parser.add_argument(
    "-f", "--folders", help="Path to folders containing images for training", type=str, nargs="+", required=True
)
parser.add_argument("--z-size", help="Latent space", type=int, default=32)
parser.add_argument("--seed", help="Random generator seed", type=int)
parser.add_argument("--n-samples", help="Max number of samples", type=int, default=-1)
parser.add_argument("-bs", "--batch-size", help="Batch size", type=int, default=64)
parser.add_argument("--learning-rate", help="Learning rate", type=float, default=1e-4)
parser.add_argument("--n-epochs", help="Number of epochs", type=int, default=25)
parser.add_argument("--verbose", help="Verbosity", type=int, default=1)
parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
args = parser.parse_args()


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def save_reconstructed_images(recon_images, epoch):
    save_image(recon_images.cpu(), f"training-images/reconstructed{epoch}.jpg")


def save_noisy_images(recon_images, epoch):
    save_image(recon_images.cpu(), f"training-images/noisy{epoch}.jpg")


def save_raw_images(recon_images, epoch):
    save_image(recon_images.cpu(), f"training-images/raw{epoch}.jpg")


def add_noise(inputs, noise_factor=0.3):
    gaussian = AddGaussianNoise(0.01, 0.03)
    inputs = gaussian(inputs)
    noisy = inputs + torch.randn_like(inputs) * noise_factor
    noisy = torch.clip(noisy, 0., 1.)
    return noisy


def train(model, dataloader, dataset_size, device):
    model.train()
    running_loss = 0.0
    counter = 0
    noise_factor = 0.3
    train_loss = 0
    for i, data in tqdm(enumerate(dataloader), total=int(dataset_size / dataloader.batch_size)):
        counter += 1
        data = data[0]
        noisy_image = add_noise(data, noise_factor)

        # pil_raw_image = to_pil_image(noisy_image[0])
        # plt.imshow(pil_raw_image)
        # plt.show()

        obs = noisy_image.to(device)
        target_obs = data.to(device)
        model.optimizer.zero_grad()

        predicted_obs = model.forward(obs)

        # pil_transformed_image = to_pil_image(predicted_obs[0])
        # plt.imshow(pil_transformed_image)
        # plt.show()

        loss = F.mse_loss(predicted_obs, target_obs)

        loss.backward()
        train_loss += loss.item()
        model.optimizer.step()
    train_loss = running_loss / counter
    return train_loss


def validate(model, dataloader, dataset, device):
    model.eval()
    running_loss = 0.0
    counter = 0
    val_loss = 0
    val_loss1 = 0
    noise_factor = 0.3
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
            counter += 1

            data = data[0]
            noisy_image = add_noise(data, noise_factor)
            obs = noisy_image.to(device)
            target_obs = data.to(device)
            predicted_obs = model.forward(obs)
            loss = F.mse_loss(predicted_obs, target_obs)
            val_loss += loss.item()
            # loss1 = F.mse_loss(predicted_obs, target_obs, reduction="none")
            # loss1 = loss1.sum(dim=[1,2,3]).mean(dim=[0])
            # val_loss1 += loss1
            # print(f'val_loss: {val_loss}')
            # print(f'val_loss1: {val_loss1}')

            # save the last batch input and output of every epoch
            if i == int(len(dataset) / dataloader.batch_size) - 1:
                recon_images = predicted_obs
                # a = noisy_image[0].permute(1, 2, 0).cpu().numpy()
                # cv2.imshow("recon", a)
                # pil_raw_image = to_pil_image(noisy_image[0])
                # plt.imshow(pil_raw_image)
                # plt.show()
                # b = recon_images[0].permute(1, 2, 0).cpu().numpy()
                # cv2.imshow("recon", b)
                # pil_transformed_image = to_pil_image(recon_images[0])
                # plt.imshow(pil_transformed_image)
                # plt.show()

                # enc = model.encode(a)
                # recon = model.decode(enc)[0]
                # cv2.imshow("enc/dec", a)
                # pil_recon_image = to_pil_image(recon)
                # plt.imshow(pil_recon_image)
                # plt.show()

    #val_loss = running_loss / counter
    return val_loss, recon_images, noisy_image, target_obs


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
    save_image(recon_images.cpu(), f"training-images/single{epoch}.jpg")



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






####################################################################

# pull_and_convert_image()

model = Autoencoder(z_size=args.z_size, learning_rate=args.learning_rate).to(device)
# model = Autoencoder(image_channels=3,
#                     base_channel_size=32,
#                     latent_dim=32,
#                     flatten_size=6144,
#                     learning_rate=0.0001).to(device)
print(model)

dataloader = DataLoader("images", 32)
writer = SummaryWriter(log_dir='runs')

# set the learning parameters
lr = 0.001
epochs = 100
batch_size = 8
best_loss = np.inf
ae_id = int(time.time())
save_path = "logs/ae-{}_{}.pkl".format(args.z_size, ae_id)
best_model_path = "logs/ae-{}_{}_best.pkl".format(args.z_size, ae_id)
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# we're using Adam optimizer here again
optimizer = optim.Adam(model.parameters(), lr=lr)

# Using Binary Cross Entropy loss for the
criterion = nn.BCELoss(reduction='sum')
# a list to save all the reconstructed images in PyTorch grid format
grid_images = []

train_loss = []
valid_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch + 1} of {epochs}")

    train_epoch_loss = train(
        model, dataloader.train_dataloader(), len(dataloader.train_dataset()), device
    )
    valid_epoch_loss, recon_images, noisy_images, raw_images = validate(
        model, dataloader.val_dataloader(), dataloader.val_dataset(), device
    )

    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)

    # save the reconstructed images from the validation loop
    save_reconstructed_images(recon_images, epoch + 1)
    #save_noisy_images(noisy_images, epoch + 1)
    #save_raw_images(raw_images, epoch + 1)
    # convert the reconstructed images to PyTorch image grid format
    image_grid = make_grid(recon_images.detach().cpu())
    grid_images.append(image_grid)

    writer.add_image('reconstructed_images', image_grid, epoch)
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

    irweazle = pull_and_convert_image()
    irweazle = irweazle.to(device)
    encoded_image = model.encode_forward(irweazle)
    something = model.decode_forward(encoded_image)
    #something = model.forward(some_image)
    something = something.to(device)
    save_single_reconstructed_images(something, epoch)
    print('something')


    # ##### this bit is working
    # image_idx = np.random.randint(len(dataloader.val_dataset()))
    # img = dataloader.val_dataset()[image_idx]
    # some_image = img[0].unsqueeze(0).to(device)
    # encoded_image = model.encode_forward(some_image)
    # something = model.decode_forward(encoded_image)
    # #something = model.forward(some_image)
    # something = something.to(device)
    # save_single_reconstructed_images(something, epoch)
    # print('something')
    # ##### end working



    # image_idx = np.random.randint(n_samples)
    # image = cv2.imread(images[image_idx])
    # r = ROI
    # im1 = image[int(r[1]) : int(r[1] + r[3]), int(r[0]) : int(r[0] + r[2])]
    # # Resize if needed
    # if ROI[2] != INPUT_DIM[1] or ROI[3] != INPUT_DIM[0]:
    #     im1 = cv2.resize(image, (INPUT_DIM[1], INPUT_DIM[0]), interpolation=cv2.INTER_AREA)
    # encoded = model.encode(im1)
    # reconstructed_image = model.decode(encoded)[0]
    # #PIL_image = Image.fromarray(np.uint8(cm.gist_earth(reconstructed_image)*255))
    # PIL_image_raw = Image.fromarray(im1.astype('uint8'), 'RGB')
    # PIL_image1 = Image.fromarray(reconstructed_image.astype('uint8'), 'RGB')
    #
    # # Plot reconstruction
    # cv2.imshow("Original", image)
    # cv2.imshow("Cropped", im1)
    # cv2.imshow("Reconstruction", reconstructed_image)
    # cv2.waitKey(1)
    # #cv2.imshow("PIL_image", PIL_image)
    # #cv2.imshow("PIL_image1", PIL_image1)
    #
    # plt.imshow(PIL_image_raw)
    # plt.axis('off')
    # plt.show()
    #
    # plt.imshow(PIL_image1)
    # plt.axis('off')
    # plt.show()
    #
    # blah = transforms.ToTensor()(im1).unsqueeze_(0).to(device)
    # blag = model.forward(blah)
    # blaa = to_pil_image(blag[0])
    # plt.imshow(blaa)
    # plt.show()
    # #blaf = Image.fromarray(blag.astype('uint8'), 'RGB')


    ##########################################


writer.flush()
writer.close()
torch.save(model.state_dict(), 'cnn_vae-32-dict.pt')
torch.save(model, 'cnn_vae-32.pt')
data = {
    "z_size": model.z_size,
    "learning_rate": model.learning_rate,
    "input_dimension": model.input_dimension
}

torch.save({"state_dict": model.state_dict(), "data": data}, save_path)
