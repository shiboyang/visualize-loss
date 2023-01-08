# @Time    : 2023/1/6 下午4:23
# @Author  : Boyang
# @Site    : 
# @File    : main.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from data import get_dataset_loader
from model import Model

batch_size = 6000
input_size = 28
learning_rate = 0.08
epochs = 100
log_interval = 10


def train(train_loader, model, optimizer, loss_func, device, epoch):
    model.train()
    vis_linear_output = []
    for batch_idx, (data, target) in enumerate(train_loader, start=1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        b = data.shape[0]
        output = model(data.view(b, -1))
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print("Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch,
                                                                           batch_idx * batch_size,
                                                                           len(train_loader.dataset),
                                                                           100.0 * batch_idx / len(train_loader),
                                                                           loss.item()))
        if epoch % 3 == 0:
            vis_linear_output.append(model.vis_linear_output)
    if len(vis_linear_output) > 1:
        vis_data = torch.cat(vis_linear_output, 0)
        x = vis_data[:, 0].tolist()
        y = vis_data[:, 1].tolist()
        plt.scatter(x, y)
        plt.title("Train")
        plt.show()


def test(test_loader, model, device, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    vis_linear_output = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            b = data.shape[0]
            output = model(data.view(b, -1))
            total_loss += F.cross_entropy(output, target, size_average=False).item()
            predict = output.max(dim=1, keepdim=True)[1]
            correct += (predict == target.view_as(predict)).sum().item()

            if epoch % 10 == 0:
                vis_linear_output.append(model.vis_linear_output)

    num_data = len(test_loader.dataset)
    total_loss /= num_data
    print("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
        total_loss, correct, num_data, 100.0 * correct / num_data
    ))
    if len(vis_linear_output) > 1:
        vis_data = torch.cat(vis_linear_output, 0)
        x = vis_data[:, 0].tolist()
        y = vis_data[:, 1].tolist()
        plt.scatter(x, y)
        plt.title("Test")
        plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = get_dataset_loader(batch_size, True)
    test_loader = get_dataset_loader(batch_size, False)

    model = Model(3, 28, 10).to(device)

    cross_entropy_loss = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(),
                          learning_rate,
                          0.9)
    for i in range(1, epochs + 1):
        train(train_loader, model, optimizer, cross_entropy_loss, device, i)
        test(test_loader, model, device, i)


if __name__ == '__main__':
    main()
