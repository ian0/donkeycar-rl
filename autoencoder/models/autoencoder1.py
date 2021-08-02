import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from autoencoder.models.encoder import Encoder
from autoencoder.models.decoder import Decoder


class Print(nn.Module):
    """"
    # Use this class to find the shape for linear layer doing a forward pass
    # Just add it as the last layer
    ## https://discuss.pytorch.org/t/linear-layer-input-neurons-number-calculation-after-conv2d/28659
    """

    def forward(self, x):
        print(x.size())
        return x


class Autoencoder(nn.Module):

    def __init__(self, image_channels=3, base_channel_size=32, latent_dim=32, flatten_size=6144, learning_rate=0.0001):
        super(Autoencoder, self).__init__()
        self.image_channels = image_channels
        self.c_hid = base_channel_size
        self.latent_dim = latent_dim
        self.flatten_size = flatten_size
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Creating encoder and decoder
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

        self.shape_before_flatten = self.encoder(torch.ones((1,) + (3, 80, 160))).shape[1:]
        flatten_size = int(np.prod(self.shape_before_flatten))

        self.encode_linear = nn.Linear(flatten_size, self.latent_dim)
        self.decode_linear = nn.Linear(self.latent_dim, flatten_size)

        self.decoder = nn.Sequential(
            nn.Conv2d(8 * self.c_hid, 4 * self.c_hid, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(4 * self.c_hid, 2 * self.c_hid, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(2 * self.c_hid, self.c_hid, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(self.c_hid, self.image_channels, kernel_size=4, stride=2),
            nn.Sigmoid()  # Scaled between 0 and 1
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)


    def forward(self, z):
        z = self.linear(z)
        z = z.reshape(z.shape[0], 256, 3, 8)
        z = self.net
        return z

    def encode_forward(self, z):
        # h = self.encoder(z).reshape(z.size(0), -1)
        # return self.encode_linear(h)
        z = self.encoder(z)
        z = self.encode_linear(z)
        return z

    def decode_forward(self, z):
        h = self.decode_linear(z).reshape((z.size(0),) + self.shape_before_flatten)
        return self.decoder(h)
        # z = self.linear(z)
        # z = z.reshape(z.shape[0], 256, 3, 8)
        # z = self.decoder(z)
        # return z

    def forward(self, input_image):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        return self.decode_forward(self.encode_forward(input_image))
        # z = self.encoder(x)
        # x_hat = self.decoder(z)
        # return x_hat

    def get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x, _ = batch  # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss
