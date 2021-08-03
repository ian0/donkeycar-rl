import torch
import cv2
from PIL import Image
from torchvision import transforms
import random

# from vae import VAE
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils

from autoencoder.dataloader.dataloader import DataLoader
from autoencoder.models.autoencoder import Autoencoder
import torchvision.transforms.functional as F



img_path = '../images/val/road/10604_cam_image_array_.jpg'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

image = cv2.imread(img_path)  # numpy array format (H, W, C=3), channel order (B, G, R)
#cv2.imshow('orig', image)
im = cv2.resize(image, (160, 80), interpolation=cv2.INTER_AREA)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.imshow(im)
plt.show()

print(f'image.shape: {image.shape}')  # (H，W，3）



observation = np.transpose(im, (2, 0, 1))
observation = torch.tensor(observation.copy(), dtype=torch.float)
print(observation.shape)
transforms = transforms.Compose(
    [transforms.Normalize(0, 255)]
)
observation = transforms(observation)
print(observation.shape)

img_tensor = torch.unsqueeze(observation, 0)
print(img_tensor.shape)
img_tensor = img_tensor.to(device)




model_path = "../../trained-models/autoencoder/ae-32_1627920759_best.pkl"
model = Autoencoder.load(model_path).to(device)
model.eval()
try:
    encoded = model.encode_forward(img_tensor)
    decoded = model.decode_forward(encoded)[0]

except AssertionError as error:
    print(error)

decoded_image = decoded.permute(1, 2, 0).detach().cpu().numpy()
plt.imshow(decoded_image)
plt.show()


# image2 = Image.open(img_path)  # PIL's JpegImageFile format (size=(W,H))
# print(image2.size)  # (W，H）
# img2_tensor = tran(image2)
# print(img2_tensor.size())
