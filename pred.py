from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models.resnet import BasicBlock, conv1x1


class RotNetW_(nn.Module):
    def __init__(self, groups=1, width_per_group: int = 64) -> None:
        super().__init__()
        self.groups = groups
        self.base_width = width_per_group
        replace_stride_with_dilation = [False, False, False]
        block = BasicBlock
        layers = [2, 2, 2, 2]
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv2d(
            8, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )

        self.layer5 = nn.Sequential(
            block(inplanes=512, planes=512),
            block(
                inplanes=512,
                planes=360,
                downsample=nn.Sequential(
                    nn.Conv2d(512, 360, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(360),
                ),
            ),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.layer5(x)
        return x

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)


class RotNetW:
    def __init__(self, weights_path="./models/last.pt"):
        assert weights_path is not None, "weights_path is None"
        if isinstance(weights_path, str):
            assert Path(weights_path).exists(), "模型权重路径不存在"
        if isinstance(weights_path, Path):
            assert weights_path.exists(), "模型权重路径不存在"
        self.model = RotNetW_()
        # 加载权重
        self.model.load_state_dict(torch.load(weights_path))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.baseCope = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
            ]
        )
        num_class = 360
        self.num2Angle = list(range(1, num_class + 1))

    def get_labels(self, y):
        return [self.num2Angle[int(num)] for num in y]

    def pred(self, center_path, border_path):
        # 路径不存在抛出异常
        if isinstance(center_path, Path) and isinstance(border_path, Path):
            assert center_path.exists() and border_path.exists(), "图片路径不存在"
        if isinstance(center_path, str) and isinstance(border_path, str):
            assert (
                Path(center_path).exists() and Path(border_path).exists()
            ), "图片路径不存在"

        center = Image.open(center_path).convert("RGBA")
        border = Image.open(border_path).convert("RGBA")
        center = self.baseCope(center).to(self.device)
        border = self.baseCope(border).to(self.device)
        imgs = torch.cat([center, border], dim=0).unsqueeze(0)
        self.model.eval()
        preds = self.get_labels(self.model(imgs).argmax(dim=-1))
        return preds[0]

