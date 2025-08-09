import os

import dill
import numpy as np
import torch
from torch.utils.data import Dataset
from ucimlrepo import fetch_ucirepo


class MyDataSet(Dataset):
    def __init__(self, x, y):
        self.x = [x]  # 初始化时将输入的x作为第一个元素
        self.y = [y]  # 初始化时将输入的y作为第一个元素
        self.indices = None
        self.dataset = None

    def __len__(self):
        # 计算所有批次数据的总长度
        return sum(len(item) for item in self.x)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.get_one_item(idx[i]) for i in range(len(idx))]
        return self.get_one_item(idx)

    def get_one_item(self, idx):
        # 遍历所有批次，找到索引对应的具体数据
        cumulative_length = 0
        for i in range(len(self.x)):
            batch_length = len(self.x[i])
            if idx < cumulative_length + batch_length:
                # 计算在当前批次中的相对索引
                local_idx = idx - cumulative_length
                return self.x[i][local_idx], self.y[i][local_idx]
            cumulative_length += batch_length

        # 如果索引超出范围，抛出异常
        raise IndexError("Index out of range")

    def append(self, x, y):
        # 确保x和y长度一致
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        self.x.append(x)
        self.y.append(y)

    def get_raw(self):
        return self.x[0], self.y[0]

    def link(self, indices, parent):
        self.indices = indices
        self.dataset = parent

    def sub2init(self, indices):
        x = self.x[0][indices]
        y = self.y[0][indices]
        subset = MyDataSet(x, y)
        subset.link(indices, self)
        return subset


def get_uci_dataset(uci_id, train_ratio=0.6):
    path = './cache'
    os.makedirs(path, exist_ok=True)
    data_path = os.path.join(path, f'uci_{uci_id}.cache')
    if os.path.exists(data_path):
        x, y, c = load_obj(data_path)
    else:
        dataset = fetch_ucirepo(id=uci_id)
        x = dataset.data.features.values
        y = dataset.data.targets.values.reshape(-1)

        unique_y = np.unique(y)
        c = len(unique_y)
        y_mapping = {val: idx for idx, val in enumerate(unique_y)}
        y = np.array([y_mapping[val] for val in y])
        save_obj((x, y, c), data_path)

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    f = x.shape[1]

    dataset = MyDataSet(x, y)
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataset = dataset.sub2init(train_dataset.indices)
    test_dataset = dataset.sub2init(test_dataset.indices)

    return train_dataset, test_dataset, c, f


def save_obj(x, path):
    with open(path, 'wb') as f:
        dill.dump(x, f)


def load_obj(path):
    with open(path, 'rb') as f:
        return dill.load(f)
