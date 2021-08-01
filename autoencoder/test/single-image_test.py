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

from src.vae.dataloader.dataloader import DataLoader
from src.vae.model import CNN_VAE
import torchvision.transforms.functional as F



img_path = '../images/val/road/10604_cam_image_array_.jpg'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

image = cv2.imread(img_path)  # numpy array format (H, W, C=3), channel order (B, G, R)
#cv2.imshow('orig', image)
im = cv2.resize(image, (160, 80), interpolation=cv2.INTER_AREA)
plt.imshow(im)
plt.show()

print(f'image.shape: {image.shape}')  # (H，W，3）
tran = transforms.ToTensor()  # Convert the numpy array or PIL.Image read image to (C, H, W) Tensor format and /255 normalize to [0, 1.0]
img_tensor = tran(image)
img_tensor.to(device)
print(f'1. img_tensor.size(): {img_tensor.size()}')  # (C,H, W), channel order (B, G, R)

#img_tensor = F.resize(img_tensor, (80, 160))

# img_tensor = torch.nn.functional.interpolate(img_tensor, size=(80, 160, 3), mode='bilinear')
print(f'2. img_tensor.size(): {img_tensor.size()}')

img_tensor = img_tensor.unsqueeze(0)
print(f'3. img_tensor.size(): {img_tensor.size()}')


model_path = "../cnn_vae-32-dict.pt"
vae_model = CNN_VAE().to(device)
vae_model.load_state_dict(torch.load(model_path))
vae_model.to(device)
vae_model.eval()
try:
    encoded, _, _ = vae_model.encode(img_tensor)
    decoded = vae_model.decode(encoded)[0]

except AssertionError as error:
    print(error)

#cv2.imshow("Reconstruction", decoded.permute(1, 2, 0).detach().cpu().numpy())
decoded_image = decoded.permute(1, 2, 0).detach().cpu().numpy()
plt.imshow(decoded_image)
plt.show()


# image2 = Image.open(img_path)  # PIL's JpegImageFile format (size=(W,H))
# print(image2.size)  # (W，H）
# img2_tensor = tran(image2)
# print(img2_tensor.size())
