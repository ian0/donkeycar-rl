from __future__ import absolute_import, division, print_function

import os

import numpy as np

ROBOT = "donkey"

# ============ DonkeyCar Config ================== #
# Raw camera input

CAMERA_HEIGHT = 120
CAMERA_WIDTH = 160

MARGIN_TOP = CAMERA_HEIGHT // 3

# ============ End of DonkeyCar Config ============ #

# Camera max FPS
FPS = 40

# Uncomment to use with the webcam:
# PICAMERA_RESOLUTION = (CAMERA_HEIGHT, CAMERA_WIDTH)

# Region Of Interest
# r = [margin_left, margin_top, width, height]
ROI = [0, MARGIN_TOP, CAMERA_WIDTH, CAMERA_HEIGHT - MARGIN_TOP]

# Fixed input dimension for the autoencoder
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 80
N_CHANNELS = 3
RAW_IMAGE_SHAPE = (CAMERA_HEIGHT, CAMERA_WIDTH, N_CHANNELS)
INPUT_DIM = (IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS)

# Arrow keys, used by opencv when displaying a window
UP_KEY = 82
DOWN_KEY = 84
RIGHT_KEY = 83
LEFT_KEY = 81
ENTER_KEY = 10
SPACE_KEY = 32
EXIT_KEYS = [113, 27]  # Escape and q
S_KEY = 115  # S key




IMAGES_PATH = '/data/roads'
BASE_IMAGE_WIDTH = 160
BASE_IMAGE_HEIGHT = 120
BASE_IMAGE_SHAPE = (120, 160, 3)

CROPPED_IMAGE_LEFT = 0
CROPPED_IMAGE_RIGHT = 160
CROPPED_IMAGE_BOTTOM = 40
CROPPED_IMAGE_TOP = 120
CROPPED_IMAGE_SHAPE = (80, 160, 3)

LATENT_SIZE=32









# """VAE Config Data in JSON"""
#
# CONFIG = {
#     "data": {
#         "images_path": "images",
#         "image_width": 160,
#         "image_height": 80,
#         "load_with_info": True
#     },
#     "train": {
#         "batch_size": 1,
#         "buffer_size": 1000,
#         "epoches": 20,
#         "val_subsplits": 5,
#         "optimizer": {
#             "type": "adam"
#         },
#         "metrics": ["accuracy"]
#     },
# }
