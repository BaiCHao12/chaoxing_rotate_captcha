# 基于imagededup去掉目录中相似度较高的图像
# 对于完全一致的图片无法去重
import os
import shutil
from pathlib import Path

from imagededup.methods import PHash

ROOT_PATH = Path.cwd()

if __name__ == "__main__":
    phasher = PHash()

    # 待处理文件夹
    src_img_path = str(ROOT_PATH.joinpath("labeled"))
    # 结果存储文件夹
    target_img_path = str(ROOT_PATH.joinpath("imgs").joinpath("center"))

    # 查找重复图像
    duplicates = phasher.find_duplicates(image_dir=src_img_path)

    # 存储已经复制的图片
    copied_images = set()
    for img_name, duImgs_name in duplicates.items():
        if img_name not in copied_images:
            # 选择其中一张图像进行复制
            src_path = os.path.join(src_img_path, img_name)
            dst_path = os.path.join(target_img_path, img_name)
            shutil.copy(src_path, dst_path)
            # 将复制的图像添加到已复制的集合中
            copied_images.add(img_name)
            # 将当前图片中相似的图片添加至已复制集合，避免二次复制
            for duImg_name in duImgs_name:
                copied_images.add(duImg_name)

    # 筛选不重复的图像
    unique_images = set(duplicates.keys()) - set(sum(duplicates.values(), []))
    # 将不重复的图像复制到目标文件夹
    for img_name in unique_images:
        src_path = os.path.join(src_img_path, img_name)
        dst_path = os.path.join(target_img_path, img_name)
        shutil.copy(src_path, dst_path)
