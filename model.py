# @Time    : 2023/1/6 下午5:58
# @Author  : Boyang
# @Site    : 
# @File    : model.py
# @Software: PyCharm
import torch
from torch import nn, Tensor
from collections import OrderedDict


class Model(nn.Module):
    def __init__(self, num_layers, input_size, num_class, device, use_l2=False, l2_scale=25):
        super(Model, self).__init__()
        self.input_size = input_size
        self.num_class = num_class
        self.feature = self._make_layers(num_layers)

        self.visualized_linear = nn.Linear(28 * 28, 2)

        self.classifier = nn.Linear(2, self.num_class, bias=False)

        self.vis_linear_output = []
        self.use_l2 = use_l2
        self.l2_scale = torch.as_tensor(l2_scale).to(device)

    def forward(self, inputs: Tensor):
        out = self.feature(inputs)
        out = self.visualized_linear(out)
        self.vis_linear_output = out.clone()

        if self.use_l2:
            out = out /  out.sum(dim=1).view(-1, 1) * self.l2_scale

        out = self.classifier(out)

        return out

    def _make_layers(self, num_layers):
        layers = OrderedDict()
        for i in range(num_layers):
            layers[f"linear{i}"] = nn.Linear(self.input_size * self.input_size, 28 * 28)
            layers[f"relu{i}"] = nn.ReLU(True)

        return nn.Sequential(layers)
