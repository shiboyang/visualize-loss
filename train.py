# @Time    : 2023/1/11 上午10:37
# @Author  : Boyang
# @Site    : 
# @File    : train.py
# @Software: PyCharm
import torch
from visualize import visualize_feature
from loss import RingLoss

ring_loss = RingLoss(10, 1)


def train(train_loader, model, optimizer, loss_func, device, epoch, batch_size, num_class):
    model.train()
    vis_linear_output = []
    vis_linear_target = []
    for batch_idx, (data, target) in enumerate(train_loader, start=1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        b = data.shape[0]
        output = model(data.view(b, -1))
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print("\rTrain epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch,
                                                                             batch_idx * batch_size,
                                                                             len(train_loader.dataset),
                                                                             100.0 * batch_idx / len(train_loader),
                                                                             loss.item()), end="")
        if epoch % 3 == 0:
            vis_linear_output.append(model.vis_linear_output)
            vis_linear_target.append(target)

    if len(vis_linear_output) > 1:
        vis_data = torch.cat(vis_linear_output, 0)
        vis_target = torch.cat(vis_linear_target, 0)
        visualize_feature(vis_data, vis_target, num_class, "Train")

    print("")


def test(test_loader, model, device, epoch, loss_func, num_class):
    model.eval()
    total_loss = 0
    correct = 0
    vis_linear_output = []
    vis_linear_target = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            b = data.shape[0]
            output = model(data.view(b, -1))
            total_loss += loss_func(output, target, size_average=False).item()
            predict = output.max(dim=1, keepdim=True)[1]
            correct += (predict == target.view_as(predict)).sum().item()

            if epoch % 10 == 0:
                vis_linear_output.append(model.vis_linear_output)
                vis_linear_target.append(target)

    num_data = len(test_loader.dataset)
    total_loss /= num_data
    print("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
        total_loss, correct, num_data, 100.0 * correct / num_data
    ))
    if len(vis_linear_output) > 1:
        vis_data = torch.cat(vis_linear_output, 0)
        vis_target = torch.cat(vis_linear_target, 0)
        visualize_feature(vis_data, vis_target, num_class, "Test")
