# @Time    : 2023/1/6 下午4:23
# @Author  : Boyang
# @Site    : 
# @File    : main.py
# @Software: PyCharm
import torch
import torch.optim as optim

from data import get_dataset_loader
from loss import CrossEntropyLoss
from model import Model
from train import train, test

batch_size = 600
input_size = 28
learning_rate = 0.001
epochs = 500
log_interval = 10
num_class = 10


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = get_dataset_loader(batch_size, True)
    test_loader = get_dataset_loader(batch_size, False)

    model = Model(3, 28, num_class, device, use_l2=True, l2_scale=25).to(device)

    cross_entropy_loss = CrossEntropyLoss()
    loss_func = cross_entropy_loss

    optimizer = optim.SGD(model.parameters(),
                          learning_rate,
                          0.9)
    for i in range(1, epochs + 1):
        train(train_loader, model, optimizer, loss_func, device, i, batch_size, num_class)
        test(test_loader, model, device, i, loss_func, num_class)


if __name__ == '__main__':
    main()
