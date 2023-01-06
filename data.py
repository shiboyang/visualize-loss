# @Time    : 2023/1/6 下午4:11
# @Author  : Boyang
# @Site    : 
# @File    : data.py
# @Software: PyCharm
import torchvision
import torch.utils.data as _data
import matplotlib.pyplot as plt

train_dataset = torchvision.datasets.MNIST("./data", train=True,
                                           transform=torchvision.transforms.ToTensor(),
                                           download=False)
test_dataset = torchvision.datasets.MNIST("./data", train=False,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=False)


def get_dataset_loader(batch_size, train):
    dataset = train_dataset if train else test_dataset

    return _data.DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True,
                            num_workers=1,
                            pin_memory=True)


if __name__ == '__main__':
    for images, label in get_dataset_loader(3, True):
        batch = images.shape[0]
        for i in range(batch):
            img = images[i].view(28, 28, 1)
            print(img.type)
            plt.imshow(img, interpolation='nearest')
            plt.title(label[i].item())
            plt.show()
        break
