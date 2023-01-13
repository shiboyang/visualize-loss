# @Time    : 2023/1/9 上午10:42
# @Author  : Boyang
# @Site    : 
# @File    : visualize.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch import Tensor
import torch

COLORS = list(mcolors.TABLEAU_COLORS.values())


def visualize_feature(data: Tensor, target: Tensor, num_class, title=""):
    assert data.shape[0] == target.shape[0]
    figure, ax = plt.subplots()
    data = torch.cat([data, target.view(-1, 1)], dim=1)
    for i in range(num_class):
        mask = data[:, 2] == i
        x = data[mask][:, 0].tolist()
        y = data[mask][:, 1].tolist()
        ax.scatter(x, y, c=COLORS[i], label=str(i), marker=".")

    ax.set_title(title)
    ax.legend()
    plt.show()
