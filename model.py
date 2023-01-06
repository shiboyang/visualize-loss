# @Time    : 2023/1/6 下午5:58
# @Author  : Boyang
# @Site    : 
# @File    : model.py
# @Software: PyCharm
from torch import nn, Tensor
from collections import OrderedDict


class Model(nn.Module):
    def __init__(self, num_layers, input_size, num_class):
        super(Model, self).__init__()
        self.input_size = input_size
        self.num_class = num_class
        self.feature = self._make_layers(num_layers)

    def forward(self, inputs: Tensor):
        self.feature(inputs)

    def _make_layers(self, num_layers):
        layers = OrderedDict()
        for i in range(num_layers):
            layers[f"linear{i}"] = nn.Linear(self.input_size * self.input_size, 28 * 28)
            layers[f"relu{i}"] = nn.ReLU(True)

        layers["liner-vis"] = nn.Linear(28 * 28, 2, bias=False)
        layers["classifier"] = nn.Linear(2, self.num_class)

        return nn.Sequential(layers)
