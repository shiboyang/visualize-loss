# @Time    : 2023/1/9 上午10:42
# @Author  : Boyang
# @Site    : 
# @File    : visualize.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import torch
from torch import Tensor


def visualize_linear(data: Tensor, target: Tensor, title=""):
    assert data.shape[0] == target.shape[0]
    b = data.shape[0]
    colors = torch.linspace(0, 1, b)
    colors = colors[target].squeeze().tolist()

    x = data[:, 0].tolist()
    y = data[:, 1].tolist()
    plt.scatter(x, y, c=colors)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    b = 10
    data = torch.randn((b, 2))
    target = torch.randint(10, (b, 1))
    visualize_linear(data, target)
