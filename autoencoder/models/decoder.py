import torch.nn as nn



class Decoder(nn.Module):

    def __init__(self,
                 num_input_channels : int,
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2*16*c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.Conv2d(8 * c_hid, 4 * c_hid, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(4 * c_hid, 2 * c_hid, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(2 *c_hid, c_hid, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=4, stride=2),
            nn.Sigmoid() # Scaled between 0 and 1
        )

    def forward(self, z):
        z = self.linear(z)
        z = z.reshape(z.shape[0], 256, 3, 8)
        z = self.net
        return z
