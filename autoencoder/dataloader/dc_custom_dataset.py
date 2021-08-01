# code modified from https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html
# https://imgaug.readthedocs.io/en/latest/source/examples_basics.html

from __future__ import print_function, division

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from imgaug import augmenters as iaa
from skimage import io, img_as_float32
from torch.utils.data import Dataset
from torchvision import transforms

from autoencoder.configs.config import BASE_IMAGE_SHAPE, \
    CROPPED_IMAGE_LEFT, CROPPED_IMAGE_RIGHT, CROPPED_IMAGE_BOTTOM, CROPPED_IMAGE_TOP


class DonkeyCarDataset(Dataset):

    def __init__(self, root_dir, transform=None, normalise_raw=True):

        self.root_dir = root_dir
        self.transform = transform
        self.data = os.listdir(self.root_dir)
        self.normalise_raw = normalise_raw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data[idx])
        raw_image = io.imread(img_name)
        transformed_image = io.imread(img_name)


        r = ResizeImage()
        raw_image = r(raw_image)

        if self.normalise_raw:
            n = NormaliseImage()
            raw_image = n(raw_image)

        #raw_image = torch.from_numpy(raw_image)


        #print(f'raw_image.shape: {raw_image.shape}')
        #print(f'transformed_image.shape: {transformed_image.shape}')


        #img = img.astype(np.uint64)
        # img = img_as_float64(img)
        #transformed = transformed.astyple(np.float32)

        # orig_transform = transforms.Compose([
        #     ResizeImage(),
        #     # AugmentImages(),
        #     #NormaliseImage(),
        #     ToTensor()
        # ])


        if self.transform:
            transformed_image = self.transform(transformed_image)
            # raw_image = orig_transform(raw_image)

        #raw_image = raw_image.permute(2, 0, 1)
        #print(f'raw_image.shape: {raw_image.shape}')
        #print(f'transformed_image.shape: {transformed_image.shape}')

        return raw_image, transformed_image


class NormaliseImage(object):
    """Normalise the images from 0-255 to 0-1"""

    def __call__(self, x):
        assert x.shape[-1] == 3, "Image must be in format numpy image: H x W x C"

        try:
            x = img_as_float32(x)
            x /= 255.0
        except AssertionError as ae:
            print(ae)
        return x


class ResizeImage(object):
    """Resize the image to 80 x 160 """

    def __call__(self, x):
        # check we are getting the image sizes we expect
        assert x.shape == BASE_IMAGE_SHAPE

        x = x[CROPPED_IMAGE_BOTTOM: CROPPED_IMAGE_TOP, CROPPED_IMAGE_LEFT: CROPPED_IMAGE_RIGHT]
        return x


class AugmentImages(object):

    def __call__(self, x):

        img = x.copy()

        seq = iaa.Sequential([
            iaa.Fliplr(0.5), # horizontal flips
            iaa.Crop(percent=(0, 0.1)), # random crops
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
            # Strengthen or weaken the contrast in each image.
            iaa.LinearContrast((0.75, 1.5)),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            iaa.Sometimes(0.5, iaa.MotionBlur(k=(3, 11), angle=(0, 360))),
            iaa.Sometimes(0.3, iaa.Cutout(nb_iterations=(1, 5), size=0.2, squared=False)),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            # iaa.Affine(
            #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            #     rotate=(-25, 25),
            #     shear=(-8, 8)
            # )
        ], random_order=True) # apply augmenters in random order

        a = seq.augment_image(img)
        return a



# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""
#
#     def __call__(self, x):
#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C X H X W
#         #x = x.transpose((2, 0, 1))
#         x = np.moveaxis(x, 2, 0)
#         out = x.astype(np.float32)
#         out = torch.from_numpy(out)
#
#         return out


def denormalise(x):
    x = np.moveaxis(x, 0, 2)
    x *= 255
    return x

if __name__ == "__main__":
    img_path = '/home/matthewi/project/ahhh/donkeycar-autoencoder/data/roads'
    dataset = DonkeyCarDataset(img_path)

    transformed_dataset = DonkeyCarDataset(img_path, transform=transforms.Compose([
        ResizeImage(),
        AugmentImages(),
        NormaliseImage(),
        ToTensor()
    ]))

    fig = plt.figure()
    for i in range(len(transformed_dataset)):
        image = transformed_dataset[i]
        orig = image[0]
        print(i, orig.shape)
        plt.imshow(np.moveaxis(orig.numpy(), 0, 2))
        plt.show()
        if i == 3:
            plt.show()
            break
        # transformed = image[1]
        # transformed = transformed.numpy()
        # img = denormalise(transformed)
        # print(i, img.shape)
        # plt.imshow(img)
        # plt.show()
        # if i == 3:
        #     plt.show()
        #     break
