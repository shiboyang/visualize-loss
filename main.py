# @Time    : 2023/1/6 下午4:23
# @Author  : Boyang
# @Site    : 
# @File    : main.py
# @Software: PyCharm
from data import get_dataset_loader
from model import Model

batch_size = 50
input_size = 28

train_data = get_dataset_loader(batch_size, True)
test_data = get_dataset_loader(batch_size, False)

model = Model(3, 28, 10)
