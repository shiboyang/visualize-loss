# @Time    : 2023/1/13 下午3:08
# @Author  : Boyang
# @Site    : 
# @File    : ring_loss.py
# @Software: PyCharm
import torch
import torch.optim as optim

from data import get_dataset_loader
from loss import CrossEntropyLoss, RingLoss
from model import FeatureExtract, Classifier
from visualize import visualize_feature

batch_size = 600
input_size = 28
learning_rate = 0.05
epochs = 500
log_interval = 10
num_class = 10


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # data
    train_loader = get_dataset_loader(batch_size, True)
    test_loader = get_dataset_loader(batch_size, False)
    # model
    feature_extract = FeatureExtract(3, input_size, 2, use_l2norm=False, l2_scale=10, device=device).to(device)
    classifier = Classifier(2, num_class).to(device)

    ringloss_func = RingLoss(10., 0.)
    cross_entropy_func = CrossEntropyLoss()

    # optimize
    optimizer = optim.SGD([{"params": feature_extract.parameters()},
                           {"params": classifier.parameters()},
                           {"params": ringloss_func.parameters()}],
                          lr=learning_rate,
                          momentum=0.9)

    for epoch in range(epochs):

        # train
        feature_extract.train()
        classifier.train()
        ringloss_func.train()

        visualized_output = []
        visualized_target = []
        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            b = data.shape[0]
            feature = feature_extract(data.view(b, -1))
            if epoch % 3 == 0:
                visualized_output.append(feature.clone())
                visualized_target.append(target)

            loss = ringloss_func(feature, target)
            out = classifier(feature)
            loss = loss + cross_entropy_func(out, target)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print("\rTrain epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch,
                                                                                 i * batch_size,
                                                                                 len(train_loader.dataset),
                                                                                 100.0 * i / len(train_loader),
                                                                                 loss.item()), end="")
        if len(visualized_output) > 1:
            vis_data = torch.cat(visualized_output, 0)
            vis_target = torch.cat(visualized_target, 0)
            visualize_feature(vis_data, vis_target, num_class, "Train")

        print("")

        # test
        # for i, (data, target) in enumerate(test_loader):
        #     ...


if __name__ == "__main__":
    main()
