from __future__ import absolute_import, division, print_function

import torch
from torch import nn
import torchvision.transforms as transforms
from torch.nn import functional as F
import numpy as np


class Print(nn.Module):
    """"
    # Use this class to find the shape for linear layer doing a forward pass
    # Just add it as the last layer
    ## https://discuss.pytorch.org/t/linear-layer-input-neurons-number-calculation-after-conv2d/28659
    """

    def forward(self, x):
        print(x.size())
        return x


class VAE(nn.Module):
    """
    Autoencoder, used to reduce the observation space in the donkeycar environment
    the input images are resized to height: 80, width: 160 before being sent to the
    autoencoder.

    :param z_size: (int) latent space dimension
    :param c_hid (int) Number of channels we use in the first convolutional layers.
    :param num_image_channels: (int) 3 for RGB images
    :param learning_rate: (float)
    :param flatten_size: (int)
    """

    def __init__(self, z_size, c_hid=32, num_image_channels=3, learning_rate=0.0001, flatten_size=6144):
        super(VAE, self).__init__()
        # AE input and output shapes
        self.z_size = z_size
        self.learning_rate = learning_rate
        self.image_channels = num_image_channels
        self.flatten_size = flatten_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = None
        self.decoder = None
        self.encode_linear = None
        self.decode_linear = None
        self.c_hid = c_hid

        self.build_encoder()
        self.build_decoder()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def build_encoder(self):
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
            # Print()
        )

        self.fc1 = nn.Linear(self.flatten_size, self.z_size)
        self.fc2 = nn.Linear(self.flatten_size, self.z_size)
        self.fc3 = nn.Linear(self.z_size, self.flatten_size)

    def build_decoder(self):

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8 * self.c_hid, 4 * self.c_hid, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(4 * self.c_hid, 2 * self.c_hid, kernel_size=4, stride=2),
            nn.ReLU(),
            ## error if kernel size is 4 -- why?
            nn.ConvTranspose2d(2 * self.c_hid, self.c_hid, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(self.c_hid, self.image_channels, kernel_size=4, stride=2),
            nn.Sigmoid(),
            # Print()
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), F.softplus(self.fc2(h))
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode_forward(self, input_tensor):
        """
        :param input_tensor: (torch.Tensor)
        :return: (torch.Tensor)
        """
        h = self.encoder(input_tensor)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode_forward(self, z):
        """
        :param z: (torch.Tensor)
        :return: (torch.Tensor)
        """
        z = self.fc3(z)
        z = z.view(z.shape[0], 256, 3, 8)
        z = self.decoder(z)
        return z

    def forward(self, input_image):
        z, mu, logvar = self.encode_forward(input_image)
        z = self.decode_forward(z)
        return z, mu, logvar

    def final_loss(self, bce_loss, mu, logvar):
        """
        This function will add the reconstruction loss (BCELoss) and the
        KL-Divergence.
        KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        :param bce_loss: recontruction loss
        :param mu: the mean from the latent vector
        :param logvar: log variance from the latent vector
        """
        BCE = bce_loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD


    def save(self, save_path):
        """
        Save to a pickle file.

        :param save_path: (str)
        """
        data = {
            "z_size": self.z_size,
            "learning_rate": self.learning_rate
        }

        torch.save({"state_dict": self.state_dict(), "data": data}, save_path)

    def encode_raw_image(self, image):
        # image.to(self.device)
        transform = transforms.Compose([
            transforms.Resize((80, 160)),
            transforms.ToTensor(),
        ])
        image = transform(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)
        return self.encode_forward(image)[0]

    def encode_ndarray(self, observation):
        """
        Encode the raw observation. It is a three channel ndarray of size 120 x 160
        :param observation:
        :return:
        """
        observation = np.transpose(observation, (2, 0, 1))
        with torch.no_grad():
            observation = torch.tensor(observation.copy(), dtype=torch.float)
            #print(observation.shape)
            transform = transforms.Compose(
                [transforms.Resize((80, 160)), transforms.Normalize(0, 255)]
            )
            observation = transform(observation)
            #print(observation.shape)

            img_tensor = torch.unsqueeze(observation, 0)
            #print(img_tensor.shape)
            img_tensor = img_tensor.to(self.device)
            return self.encode_forward(img_tensor)



    @classmethod
    def load(cls, load_path):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        saved_variables = torch.load(load_path, map_location=device)
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
        autoencoder = VAE(z_size=1)
    else:
        autoencoder = VAE.load(path)
    print("Dim AE = {}".format(autoencoder.z_size))
    print("PyTorch", torch.__version__)
    return autoencoder
