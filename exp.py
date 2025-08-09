import datetime
import os
import sys
import traceback
import uuid

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from data import get_uci_dataset
from enhance import Enhancer
from model import MLPClassifier


class Exp:
    def __init__(self, args):
        self.args = args
        self.device = utils.get_device(args.device)
        self.args.device = str(self.device)
        print(self.device)

        self.enhancer = None
        self.set_enhance()

        self.train_data = None
        self.test_data = None
        self.set_data()

        self.model = None
        self.criterion = None
        self.optimizer = None
        self.set_model()
        utils.print_args(self.args)

    def set_enhance(self):
        self.enhancer = Enhancer(self.args) if self.args.enhance else None

    def set_data(self):
        self.train_data, self.test_data, self.args.num_classes, self.args.input = get_uci_dataset(self.args.uci_id,
                                                                                                  train_ratio=self.args.train_ratio)

    def set_model(self):
        self.model = MLPClassifier(self.args.input,
                                   self.args.hidden,
                                   self.args.num_classes,
                                   self.args.dropout,
                                   self.args.layers).to(self.device)
        self.criterion = F.cross_entropy
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

    def exp_init(self):
        if self.enhancer:
            x_extra, y_extra = self.enhancer.generate(self.train_data)
            self.train_data.append(x_extra, y_extra)
            pass

    def train(self):
        self.model.train()

        train_loader = DataLoader(
            self.train_data,
            batch_size=self.args.batch,
            shuffle=True,
        )

        bar = tqdm(range(self.args.epochs))
        for _ in bar:
            running_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            bar.set_postfix({"Loss": running_loss / len(train_loader)})

    def test(self):
        self.model.eval()  # 设置模型为评估模式

        test_loader = DataLoader(
            self.test_data,
            batch_size=self.args.batch,
            shuffle=False
        )

        total = 0
        correct = 0

        with torch.no_grad():
            bar = tqdm(test_loader)
            for inputs, labels in bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)  # 获取预测类别

                total += labels.size(0)  # 累加总样本数
                correct += (predicted == labels).sum().item()  # 累加正确预测数
                bar.set_postfix({"correct": correct})

        # 计算准确率
        self.args.accuracy = correct / total
        print(f'Test Accuracy: {self.args.accuracy * 100:.2f}%')

    def save(self):
        path = './out'
        if not os.path.exists(path):
            os.makedirs(path)
        uid = uuid.uuid4().hex[:8]
        setting = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") + "_" + uid
        out_path = os.path.join(path, setting)
        try:
            os.makedirs(out_path)
            utils.save_args_to_json(self.args, os.path.join(out_path, 'args.json'))
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)
