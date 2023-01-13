# @Time    : 2023/1/6 下午5:58
# @Author  : Boyang
# @Site    : 
# @File    : model.py
# @Software: PyCharm
import torch
from torch import nn, Tensor
from collections import OrderedDict


def _make_linear_layer(num_layers, input_size, output_size):
    layers = OrderedDict()
    for i in range(num_layers):
        layers[f"linear{i}"] = nn.Linear(input_size ** 2, output_size ** 2, bias=True)
        layers[f"relu{i}"] = nn.ReLU(True)

    return nn.Sequential(layers)


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
            out = out / torch.sqrt((out ** 2).sum(dim=1)).view(-1, 1) * self.l2_scale

        out = self.classifier(out)

        return out


class FeatureExtract(nn.Module):
    def __init__(self, num_layers, input_size, output_size, use_l2norm, l2_scale, device):
        super(FeatureExtract, self).__init__()
        self.feature = _make_linear_layer(num_layers, input_size, input_size)
        self.visualized_linear = nn.Linear(input_size ** 2, output_size, bias=False)
        self.vis_output = None
        self.use_l2norm = use_l2norm
        self.l2_scale = torch.as_tensor(l2_scale, device=device)

    def forward(self, inputs: Tensor):
        out = self.feature(inputs)
        out = self.visualized_linear(out)
        self.vis_output = out.clone()
        if self.use_l2norm:
            out = out / torch.linalg.norm(out, dim=1, keepdim=True) * self.l2_scale

        return out


class Classifier(nn.Module):
    def __init__(self, input_size, num_class):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(input_size, num_class, bias=False)

    def forward(self, inputs):
        torch.linalg.norm(self.classifier.weight, dim=1)
        return self.classifier(inputs)

