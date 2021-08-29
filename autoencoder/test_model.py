import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.transforms.functional import crop
from PIL import Image
from torchvision.utils import save_image

plt.style.use('ggplot')

from models.autoencoder import Autoencoder
from autoencoder.models.variational_autoencoder import VAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("-ae", "--ae-path", help="Path to saved autoencoder (otherwise start from scratch)",
                    type=str, default='logs/track/vae-64_1629830899_best.pkl')
parser.add_argument(
    "-f", "--folders", help="Path to folders containing images for training", type=str, nargs="+",
    required=False
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
model = VAE.load(args.ae_path)
#args.z_size = model.z_size
model.eval()

images = []

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

for filename in os.listdir('images/track/val/track/'):
    if filename.endswith("jpg"):
        # Your code comes here such as
        #print(filename)
        pil_im = Image.open(f'images/track/val/track/{filename}')
        img = np.asarray(pil_im)
        images.append(img)



def add_noise(inputs, noise_factor=0.3):
    gaussian = AddGaussianNoise(0.05, 0.05)
    inputs = gaussian(inputs)
    noisy = inputs + torch.randn_like(inputs) * noise_factor
    noisy = torch.clip(noisy, 0., 1.)
    return noisy

n_samples = len(images)

def custom_crop(image):
    return crop(image, 40, 0, 80, 160)


def show_img(img):
    plt.figure()
    img = img.permute(1, 2, 0)
    plt.imshow(img)
    plt.show()


def save_single_reconstructed_images(recon_images, epoch):
    save_image(recon_images.cpu(), f"training-images/single{epoch}.jpg")


def pull_image():
    image_idx = np.random.randint(n_samples)
    image = images[image_idx]
    return image


def pull_and_convert_image():
    image_idx = np.random.randint(n_samples)
    # image = cv2.imread(images[image_idx])
    image =images[image_idx]
    # image.show()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(custom_crop),
        transforms.Resize((80, 160)),
    ])
    image = transform(image)
    #image = image.unsqueeze(0)
    print(image.size())
    return image


irweazle = pull_and_convert_image()
trans = transforms.ToPILImage()
plt.imshow(trans(irweazle))
plt.show()
#
# irweazle = trans(irweazle)
# irweazle.save('training-images/noisy_track_image.png')

irweazle = pull_image()
irweazle = Image.fromarray(irweazle)
encoded_image = model.encode_raw_image(irweazle)
something = model.decode_forward(encoded_image)
something = something.to(device)
save_single_reconstructed_images(something, 1001)
print('something')
