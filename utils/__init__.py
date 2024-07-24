import numpy as np
import time

class Timer:
    """用于记录多次运行时间的类。"""

    def __init__(self):
        """在 :numref:`sec_minibatch_sgd` 中定义。"""
        self.times = []  # 用于存储每次运行时间的时间戳列表
        self.start()  # 开始计时

    def start(self):
        """开始计时，将当前时间戳保存到 self.tik 中。"""
        self.tik = time.time()

    def stop(self):
        """停止计时，并将当前时间戳减去 self.tik 的差值添加到 self.times 列表中。返回最后一次运行的时间。"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """计算并返回平均时间，即 self.times 列表中所有时间之和除以时间列表的长度。"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和，即 self.times 列表中所有时间之和。"""
        return sum(self.times)

    def cumsum(self):
        """返回累积时间，即 self.times 列表中所有时间构成的累积和。"""
        return np.array(self.times).cumsum().tolist()
