# From: https://github.com/araffin/learning-to-drive-in-5-minutes/blob/master/vae/controller.py
# Orignal author: Roma Sokolkov
# VAE controller for runtime optimization.

import numpy as np

from autoencoder.models.autoencoder import Autoencoder



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
        z_size=32,
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

        self.vae = None
        self.target_vae = None

        if z_size is not None:
            # self.vae = ConvVAE(
            #     z_size=self.z_size,
            #     batch_size=self.batch_size,
            #     learning_rate=self.learning_rate,
            #     kl_tolerance=self.kl_tolerance,
            #     is_training=True,
            #     reuse=False,
            # )
            #
            # self.target_vae = A(
            #     z_size=self.z_size, batch_size=1, is_training=False, reuse=False
            # )
            self.autoencoder = Autoencoder.load('/home/matthewi/project/ahhh/donkeycar-rl/trained-models/autoencoder/ae-32_1627920759_best.pkl')

    def encode(self, observation):
        assert observation.shape == self.input_dimension, "{} != {}".format(
            observation.shape, self.input_dimension
        )
        #observation = observation.copy().astype(np.float32) / 255.0
        tensor_obs = self.autoencoder.encode_ndarray(observation)
        return self.autoencoder.encode_ndarray(observation).cpu().numpy()
        #return self.autoencoder.encode_from_raw_image(observation)


    def decode(self, arr):
        assert arr.shape == (1, self.z_size), "{} != {}".format(
            arr.shape, (1, self.z_size)
        )
        # Decode
        arr = self.target_vae.decode(arr)
        # Denormalize
        arr = (255 * np.clip(arr, 0, 1)).astype(np.uint8)
        return arr

    def save(self, path):
        self.target_vae.save(path)

    # def load(self, path):
    #     self.target_vae = ConvVAE.load(path)
    #     self.z_size = self.target_vae.z_size

    def set_target_params(self):
        params = self.vae.get_params()
        self.target_vae.set_params(params)
