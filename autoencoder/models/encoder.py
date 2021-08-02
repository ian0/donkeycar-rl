import torch.nn as nn
import torch
import numpy as np


class Print(nn.Module):
    def forward(self, x):
        print(x.size())
        return x

class Encoder(nn.Module):

    def __init__(self,
                 num_input_channels,
                 base_channel_size,
                 latent_dim,
                 flatten_size=6144):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image.
            - base_channel_size : Number of channels we use in the first convolutional layers.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=4, stride=2),  # 32x32 => 16x16
            nn.ReLU(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(2 * c_hid, 4 * c_hid, kernel_size=4, stride=2),  # 16x16 => 8x8
            nn.ReLU(),
            nn.Conv2d(4 * c_hid, 8 * c_hid, kernel_size=4, stride=2),
            nn.ReLU(),
            Print()
        )

        # shape_before_flatten = self.net(torch.ones((1,) + (3, 80, 160))).shape[1:]
        # print(f'shape_before_flatten: {shape_before_flatten}')
        # flatten_size = int(np.prod(shape_before_flatten))


        nn.Flatten(),  # Image grid to single feature vector
        nn.Linear(flatten_size, latent_dim)

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    encode = Encoder(3, 32, 32, 6144)
