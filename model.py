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

        self.visualized_linear = nn.Linear(28 * 28, 2)

        self.classifier = nn.Linear(2, self.num_class, bias=False)

        self.vis_linear_output = []

    def forward(self, inputs: Tensor):
        out = self.feature(inputs)
        out = self.visualized_linear(out)
        self.vis_linear_output = out.clone()
        out = self.classifier(out)
        return out

    def _make_layers(self, num_layers):
        layers = OrderedDict()
        for i in range(num_layers):
            layers[f"linear{i}"] = nn.Linear(self.input_size * self.input_size, 28 * 28)
            layers[f"relu{i}"] = nn.ReLU(True)

        return nn.Sequential(layers)
