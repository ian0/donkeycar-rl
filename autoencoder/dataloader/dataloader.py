import torch
import torch.utils.data
import torchvision.transforms as transforms
from torchvision import datasets as datasets


class DataLoader:
    """Data Loader class"""

    def __init__(self, image_path, batch_size):
        self.image_path = image_path
        self.batch_size = batch_size

    def train_dataset(self):
        return datasets.ImageFolder(root=F'{self.image_path}/train',
                                    transform=self.train_transformer())

    def val_dataset(self):
        return datasets.ImageFolder(root=F'{self.image_path}/val',
                                    transform=self.train_transformer())

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset(), batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset(), batch_size=self.batch_size, shuffle=True)

    @staticmethod
    def train_transformer():
        return transforms.Compose([
            transforms.Resize((80, 160)),
            transforms.ToTensor(),
        ])

    @staticmethod
    def val_transformer():
        return transforms.Compose([
            transforms.Resize((80, 160)),
            transforms.ToTensor(),
        ])
