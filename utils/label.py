import json
import tkinter as tk
from pathlib import Path
from tkinter import StringVar, messagebox

from PIL import Image, ImageTk

from pred import pred

ROOT_PATH = Path.cwd()
LABELS_PATH = ROOT_PATH / "imgs_3"  # 需要标注的图片旋转体路径
LABELS_BORDER_PATH = LABELS_PATH / "border"  # 需要标注的图片外部边框路径
OUTPUT_PATH = ROOT_PATH.joinpath("labeled_valid_3")  # 输出旋转到正确角度的验证码图片
OUTPUT_PATH.mkdir(exist_ok=True)
# 标注过的配置信息
CONFIG_PATH = LABELS_PATH.joinpath("config.json")


# LABELS_PATH = ROOT_PATH.joinpath("utils").joinpath("imgs2")  # 需要标注的图片旋转体路径
# LABELS_BORDER_PATH = LABELS_PATH.joinpath("border")  # 需要标注的图片外部边框路径
# OUTPUT_PATH = ROOT_PATH.joinpath("labeled")  # 输出旋转到正确角度的验证码图片
# OUTPUT_PATH.mkdir(exist_ok=True)
# # 标注过的配置信息
# CONFIG_PATH = LABELS_PATH.joinpath("config.json")


class ImageRotatorApp:
    def __init__(self, master):
        self.config = {}

        self.model_pred = StringVar(master=master, value="0")
        self.master = master
        self.title = "旋转验证码标注"
        self.master.title(self.title)

        # 初始化变量
        self.angle = 0
        self.image_index = 0
        # 获取路径下全部jpg文件
        self.image1_list = [i for i in LABELS_PATH.glob("*.png")]
        self.all_image_index = len(self.image1_list)

        # if CONFIG_PATH.exists():
        #     with open(CONFIG_PATH) as f:
        #         self.config = json.loads(f.read())
        #         self.image_index = len(self.config)

        self.updateWidget()
        self.createWidget()
        self.rotation_slider.set(int(self.model_pred.get()))

    def rotate_image(self, angle):
        if not hasattr(self, "angle_entry"):
            return

        self.angle = int(angle)
        self.angle_entry.delete(0, tk.END)
        self.angle_entry.insert(0, str(abs(self.angle)))

        # Rotate image1
        self.rotated_image1 = self.image1.rotate(-self.angle)
        self.rotated_image1_tk = ImageTk.PhotoImage(self.rotated_image1)
        self.canvas.itemconfig(self.image1_id, image=self.rotated_image1_tk)
        # Update image2 position to keep it centered
        self.canvas.coords(self.image1_id, self.center_x, self.center_y)

    def validate_input(self, event):
        value = self.angle_var.get()
        if not value.isdigit():
            self.angle_var.set(str(self.angle))
        else:
            self.rotation_slider.set(value)
            self.rotate_image(value)

    def adjust_angle(self, delta):
        new_angle = int(self.angle_var.get()) + delta
        if 0 <= new_angle <= 360:
            self.angle_var.set(str(new_angle))
            self.rotation_slider.set(new_angle)
            self.rotate_image(new_angle)

    def save_rotate_image(self, image_path, degrees_to_rotate, output_path):
        # 打开图像
        image = Image.open(image_path)

        # 旋转图像
        rotated_image = image.rotate(-degrees_to_rotate)

        # 保存旋转后的图像
        rotated_image.save(output_path)

        print("图像已旋转并保存到:", output_path)

    def commit_angle(self):
        angle = int(self.angle_var.get())
        nowImageCenter = self.image1_list[self.image_index]
        self.config[nowImageCenter.name] = angle
        with open(CONFIG_PATH, "w") as f:
            f.write(json.dumps(self.config))
        self.save_rotate_image(
            nowImageCenter,
            angle,
            OUTPUT_PATH.joinpath(nowImageCenter.name.replace("jpg", "png")),
        )

    def next_group(self):
        # Change images
        self.commit_angle()
        self.image_index += 1
        self.updateWidget()
        # Update canvas
        self.updateCanvas()
        self.rotation_slider.set(int(self.model_pred.get()))

    def delete(self):
        """删除当前一组验证码"""
        if messagebox.askokcancel("删除", "确定删除当前组验证码？"):
            self.all_image_index -= 1
            delete_img_center = self.image1_list[self.image_index]
            delete_img_border = self.image2_list[self.image_index]
            delete_img_center.unlink()
            delete_img_border.unlink()
            self.image_index += 1
            self.updateWidget()
            self.updateCanvas()

    def createWidget(self):
        # Create canvas
        self.canvas = tk.Canvas(
            self.master, width=self.image1.width, height=self.image1.height
        )
        self.canvas.pack()

        # 展示图片
        self.center_x = (self.image2.width - self.image1.width) // 2
        self.center_y = (self.image2.height - self.image1.height) // 2
        self.image1_id = self.canvas.create_image(
            self.center_x, self.center_y, anchor=tk.NW, image=self.image1_tk
        )
        self.image2_id = self.canvas.create_image(
            0, 0, anchor=tk.NW, image=self.image2_tk
        )
        self.bottom_frame = tk.Frame(self.master)
        self.bottom_frame.pack(pady=(0, 16), padx=16, fill="x")
        # Create rotation slider
        self.rotation_slider = tk.Scale(
            self.bottom_frame,
            from_=0,
            to=360,
            orient=tk.HORIZONTAL,
            length=360,
            command=self.rotate_image,
        )
        self.rotation_slider.pack(pady=(0, 8))

        self.row_2 = tk.Frame(self.bottom_frame)
        self.row_2.pack(pady=(0, 8), fill="x")
        self.pred_frame = tk.Frame(self.row_2)
        self.pred_frame.pack(side="left")
        self.pred_title = tk.Label(self.row_2, text="预测角度：")
        self.pred_title.pack(side="left")
        self.pred = tk.Label(self.row_2, textvariable=self.model_pred)
        self.pred.pack(side="left")

        self.input_frame = tk.Frame(self.row_2)
        self.input_frame.pack(side="right")
        # Create angle entry
        self.decrease_button = tk.Button(
            self.input_frame, text="-", command=lambda: self.adjust_angle(-1)
        )
        self.decrease_button.pack(side="left", pady=5)
        self.angle_var = tk.StringVar()
        self.angle_var.set(str(self.angle))
        validate_cmd = self.master.register(self.validate_input)
        self.angle_entry = tk.Entry(
            self.input_frame,
            textvariable=self.angle_var,
            validate="key",
            validatecommand=(validate_cmd, "%P"),
        )
        self.angle_entry.pack(side="left")
        self.increase_button = tk.Button(
            self.input_frame, text="+", command=lambda: self.adjust_angle(1)
        )
        self.increase_button.pack(side="left", pady=5)

        # 按钮组
        self.button_frame = tk.Frame(self.bottom_frame)
        self.button_frame.pack(pady=(0, 10), fill="x")
        self.commit_button = tk.Button(
            self.button_frame, text="提交", command=self.commit_angle
        )
        self.commit_button.pack(side="left")
        self.next_button = tk.Button(
            self.button_frame, text="下一组", command=self.next_group
        )
        self.next_button.pack(side="right")

        self.del_button = tk.Button(self.bottom_frame, text="删除", command=self.delete)
        self.del_button.pack(side="right")

    def updateWidget(self):
        """切换图片时更新相关控件"""
        if self.image_index >= self.all_image_index:
            messagebox.showinfo("提示", "标注结束")
            return
        self.master.title(f"{self.title} - {self.image_index+1}/{self.all_image_index}")
        # 加载图片
        self.image1 = Image.open(self.image1_list[self.image_index])
        self.image2 = Image.open(
            LABELS_BORDER_PATH / self.image1_list[self.image_index].name
        )
        pred_angle = pred(
            center_path=self.image1_list[self.image_index],
            border_path=LABELS_BORDER_PATH / self.image1_list[self.image_index].name,
        )
        self.model_pred.set(str(360 - pred_angle))

        self.image1_tk = ImageTk.PhotoImage(self.image1)
        self.image2_tk = ImageTk.PhotoImage(self.image2)

    def updateCanvas(self):
        self.canvas.itemconfig(self.image1_id, image=self.image1_tk)
        self.canvas.itemconfig(self.image2_id, image=self.image2_tk)
        self.rotation_slider.set(0)  # Reset slider
        self.angle = 0
        self.angle_entry.delete(0, tk.END)
        self.angle_entry.insert(0, str(self.angle))


if __name__ == "__main__":
    root = tk.Tk()
    ImageRotatorApp(root)
    root.mainloop()
