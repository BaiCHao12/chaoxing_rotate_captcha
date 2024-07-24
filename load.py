import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from setting import angle2Num, num2Angle, setting

DATA_PATH = Path.cwd() / "data"
CENTER_PATH = DATA_PATH / "center"

TRAIN_CENTER_PATH = CENTER_PATH / "train"
TEST_CENTER_PATH = CENTER_PATH / "test"
VALID_CENTER_PATH = CENTER_PATH / "valid"

BORDER_PATH = DATA_PATH / "border"

train_center_data = list(TRAIN_CENTER_PATH.glob("*.png"))
test_center_data = list(TEST_CENTER_PATH.glob("*.png"))
valid_center_data = list(VALID_CENTER_PATH.glob("*.png"))

transform_base = transforms.Compose(
    [
        transforms.Resize(setting.size),
        transforms.ToTensor(),
    ]
)
train_transforms = transforms.Compose(
    [
        transforms.Resize(setting.size),
        transforms.ColorJitter(0.5),
        transforms.ToTensor(),
    ]
)


class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        super().__init__()
        self.data = data
        self.transform = transform_base
        if transform:
            self.transform = transform

    def __getitem__(self, idx):
        center = Image.open(self.data[idx]).convert("RGBA")
        border = Image.open(BORDER_PATH / self.data[idx].name).convert("RGBA")
        # 从num2Angle中随机选择角度
        angle = random.choice(num2Angle)
        center = center.rotate(-angle)
        center = self.transform(center)
        border = self.transform(border)
        # 将图片合并为8通道数据
        imgs = torch.cat([center, border], dim=0)
        # 角度转换为类别数
        # 忘记编码会报错
        label = torch.tensor(angle2Num[angle])
        return imgs.to(setting.device), label.to(setting.device)

    def __len__(self):
        return len(self.data)


train_dataset = MyDataset(train_center_data * 360, train_transforms)
test_dataset = MyDataset(test_center_data * 360)
valid_dataset = MyDataset(valid_center_data * 360)

train_iter = DataLoader(train_dataset, setting.batch_size, shuffle=True)
test_iter = DataLoader(test_dataset, setting.batch_size, shuffle=True)
valid_iter = DataLoader(valid_dataset, setting.batch_size, shuffle=True)
