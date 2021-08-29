# From: https://github.com/araffin/learning-to-drive-in-5-minutes/blob/master/vae/controller.py
# Orignal author: Roma Sokolkov
# VAE controller for runtime optimization.

import numpy as np

from autoencoder.models.autoencoder import Autoencoder
from autoencoder.models.variational_autoencoder import VAE
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class AEController:
    """
    Wrapper to manipulate a VAE.
    :param z_size: (int) latent space dimension
    :param input_dimension: ((int, int, int)) input dimension
    :param learning_rate: (float)
    :param kl_tolerance: (float) Clip the KL loss
        max_kl_loss = kl_tolerance * z_size
    :param batch_size: (int)
    :param normalization_mode: (str)
    """

    def __init__(
        self,
        path,
        z_size=64,
        input_dimension=(120, 160, 3),
        learning_rate=0.0001,
        kl_tolerance=0.5,
        batch_size=64,
        normalization_mode="rl",
    ):
        # VAE input and output shapes
        self.z_size = z_size
        self.input_dimension = input_dimension

        # VAE params
        self.learning_rate = learning_rate
        self.kl_tolerance = kl_tolerance

        # Training params
        self.batch_size = batch_size
        self.normalization_mode = normalization_mode

        self.ae = None
        self.target_ae = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if z_size is not None:
            #self.autoencoder = Autoencoder.load('/home/matthewi/project/ahhh/donkeycar-rl/trained-models/autoencoder/ae-32_1627920759_best.pkl')
            self.ae = VAE.load(path)
            self.target_ae = VAE.load(path)

    def encode(self, observation):
        assert observation.shape == self.input_dimension, "{} != {}".format(
            observation.shape, self.input_dimension
        )
        return self.ae.encode_ndarray(observation).cpu().numpy()

    def decode(self, encoded):
        assert encoded.shape == (1, self.z_size), "{} != {}".format(
            encoded.shape, (1, self.z_size)
        )
        # Decode
        encoded_tensor = torch.from_numpy(encoded).to(self.device)
        decoded = self.target_ae.decode_forward(encoded_tensor)
        decoded = torch.squeeze(decoded)
        trans = transforms.ToPILImage()
        decoded_image = trans(decoded)
        # plt.imshow(trans(decoded))
        # plt.show()
        # # Denormalize
        # decoded = (255 * np.clip(decoded.numpy(), 0, 1)).astype(np.uint8)
        return decoded_image

    def save(self, path):
        self.target_ae.save(path)

    def set_target_params(self):
        params = self.ae.get_params()
        self.target_ae.set_params(params)
