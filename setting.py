from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Setting:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path.cwd() / "models"
    # 确保模型文件所在的目录存在，如果不存在则创建该目录及其所有必要的父目录
    model_path.mkdir(parents=True, exist_ok=True)

    is_log = True
    # 分类数
    num_class: int = 360
    # 数据增强
    size: int = 224
    # 超参数
    lr: float = 0.5
    batch_size: int = 256

    num_epochs: int = 100
    momentum: float = 0.9
    # 上升下降所占周期 默认0.3 最大0.5
    pct_start: float = 0.25


setting = Setting()
num2Angle = list(range(1, setting.num_class + 1))
angle2Num = {value: key for key, value in enumerate(num2Angle)}
