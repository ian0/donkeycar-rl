from __future__ import absolute_import, division, print_function

import cv2  # pytype: disable=import-error
import numpy as np
import torch as th
from torch import nn

from autoencoder.configs.config import INPUT_DIM, RAW_IMAGE_SHAPE, ROI

class Print(nn.Module):
    def forward(self, x):
        print(x.size())
        return x


class Autoencoder(nn.Module):
    """
    Wrapper to manipulate an autoencoder.

    :param z_size: (int) latent space dimension
    :param input_dimension: ((int, int, int)) input dimension
    :param learning_rate: (float)
    :param normalization_mode: (str)
    """

    def __init__(self, z_size, input_dimension=INPUT_DIM, learning_rate=0.0001, flatten_size=6144):
        super(Autoencoder, self).__init__()
        # AE input and output shapes
        self.z_size = z_size
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        self.flatten_size = flatten_size

        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.encoder = None
        self.decoder = None
        self.shape_before_flatten = None
        self.c_hid = 32
        self.image_channels = 3

        # Re-order
        h, w, c = input_dimension
        self._build((c, h, w))
        self.optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate)





    def _build(self, input_shape):
        # n_channels, kernel_size, strides, activation, padding=0
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(input_shape[0], 32, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 256, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     #Print()
        # )
        self.encoder = nn.Sequential(
            nn.Conv2d(self.image_channels, self.c_hid, kernel_size=4, stride=2),  # 32x32 => 16x16
            nn.ReLU(),
            nn.Conv2d(self.c_hid, 2 * self.c_hid, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(2 * self.c_hid, 4 * self.c_hid, kernel_size=4, stride=2),  # 16x16 => 8x8
            nn.ReLU(),
            nn.Conv2d(4 * self.c_hid, 8 * self.c_hid, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),  # Image grid to single feature vector
            # nn.Linear(flatten_size, latent_dim),
            # Print()
        )

        # Compute the shape doing a forward pass
        ## Use this link to calculate the size of the flattened shape:
        ## https://discuss.pytorch.org/t/linear-layer-input-neurons-number-calculation-after-conv2d/28659
        self.shape_before_flatten = self.encoder(th.ones((1,) + input_shape)).shape[1:]
        flatten_size = int(np.prod(self.shape_before_flatten))

        self.encode_linear = nn.Linear(self.flatten_size, self.z_size)
        self.decode_linear = nn.Linear(self.z_size, self.flatten_size)

        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, input_shape[0], kernel_size=4, stride=2),
        #     nn.Sigmoid(),
        #     #Print()
        # )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8 * self.c_hid, 4 * self.c_hid, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(4 * self.c_hid, 2 * self.c_hid, kernel_size=4, stride=2),
            nn.ReLU(),
            ## error is kernel size is 4 -- why?
            nn.ConvTranspose2d(2 * self.c_hid, self.c_hid, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(self.c_hid, self.image_channels, kernel_size=4, stride=2),
            nn.Sigmoid(),
            #Print()
        )
        # self.decoder = nn.Sequential(
        #     nn.Conv2d(8 * self.c_hid, 4 * self.c_hid, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(4 * self.c_hid, 2 * self.c_hid, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(2 * self.c_hid, self.c_hid, kernel_size=5, stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(self.c_hid, input_shape[0], kernel_size=4, stride=2),
        #     nn.Sigmoid()  # Scaled between 0 and 1
        # )


    def encode_forward(self, input_tensor):
        """
        :param input_tensor: (th.Tensor)
        :return: (th.Tensor)
        """
        # h = self.encoder(input_tensor).reshape(input_tensor.size(0), -1)
        # return self.encode_linear(h)
        z = self.encoder(input_tensor)#.view(input_tensor.size(0), -1)
        z = self.encode_linear(z)
        return z


    def decode_forward(self, z):
        """
        :param z: (th.Tensor)
        :return: (th.Tensor)
        """
        # h = self.decode_linear(z).reshape((z.size(0),) + self.shape_before_flatten)
        # return self.decoder(h)
        z = self.decode_linear(z)
        z = z.view(z.shape[0], 256, 3, 8)
        z = self.decoder(z)
        return z

    def forward(self, input_image):
        return self.decode_forward(self.encode_forward(input_image))

    def save(self, save_path):
        """
        Save to a pickle file.

        :param save_path: (str)
        """
        data = {
            "z_size": self.z_size,
            "learning_rate": self.learning_rate,
            "input_dimension": self.input_dimension
        }

        th.save({"state_dict": self.state_dict(), "data": data}, save_path)

    @classmethod
    def load(cls, load_path):
        device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")
        saved_variables = th.load(load_path, map_location=device)
        model = cls(**saved_variables["data"])
        model.load_state_dict(saved_variables["state_dict"])
        model.to(device)
        return model





def load_ae(path=None, z_size=None, quantize=False):
    """
    :param path: (str)
    :param z_size: (int)
    :param quantize: (bool) Whether to quantize the model or not
    :return: (Autoencoder)
    """
    # z_size will be recovered from saved model
    if z_size is None:
        assert path is not None

    # Hack to make everything work without trained AE
    if path == "dummy":
        autoencoder = Autoencoder(z_size=1)
    else:
        autoencoder = Autoencoder.load(path)
    print("Dim AE = {}".format(autoencoder.z_size))
    print("PyTorch", th.__version__)
    return autoencoder
