import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, input, hidden, num_classes, dropout, layers):
        super().__init__()
        self.norm = nn.BatchNorm1d(input)

        # 创建层列表
        self.layers = nn.ModuleList()

        # 添加第一个隐藏层（输入层到第一个隐藏层）
        self.layers.append(nn.Sequential(
            nn.Linear(input, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        ))

        # 添加中间隐藏层
        for _ in range(layers - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))

        # 输出层
        self.output_layer = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.norm(x)
        # 依次通过所有隐藏层
        for layer in self.layers:
            x = layer(x)

        # 通过输出层
        x = self.output_layer(x)
        return x
