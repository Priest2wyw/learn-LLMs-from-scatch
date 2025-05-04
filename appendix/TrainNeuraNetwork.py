#! /bin/python
import time
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class NeralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 第一个隐藏层
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 第二个隐藏层
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            # 第三个隐藏层
            torch.nn.Linear(20, num_outputs),
        )
    def forward(self, x):
        # 前向传播
        x = self.layers(x)
        return x

def count_parameters(model):
    # 计算模型参数数量
    para_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数数量：{para_count}")
    return para_count
class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_data():
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])

    y_test = torch.tensor([0, 1])

    train_ds = ToyDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, 
                        batch_size=2,  # batch_size=2表示每个batch包含2个样本
                        shuffle=True,  # shuffle=True表示每个epoch随机打乱数据
                        num_workers=0, # num_workers=0表示不使用多线程加载数据
                        drop_last=True # drop_last=True表示如果最后一个batch不满2个样本，则丢弃
                        )

    # 画出 2 维图,看看0/1类长什么样子
    import matplotlib.pyplot as plt
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=100, alpha=0.5)
    test_ds = ToyDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0)

    return train_loader, test_loader
def test_data_loader():
    """
    测试数据加载器
    
    """
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])

    y_test = torch.tensor([0, 1])

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    # 修复问题
    
    start_time = time.time()
    train_loader = DataLoader(train_ds, 
                        batch_size=2,  # batch_size=2表示每个batch包含2个样本
                        shuffle=True,  # shuffle=True表示每个epoch随机打乱数据
                        num_workers=0, # num_workers=0表示不使用多线程加载数据
                        drop_last=True # drop_last=True表示如果最后一个batch不满2个样本，则丢弃
                        )
    print(f"采用 drop_last=True, num_workers=0")
    for i, (X, y) in enumerate(train_loader):
        print(f"Batch {i+1}: {X=},{y=}")
    end_time = time.time()
    print(f"数据加载时间：{end_time - start_time:.4f}秒")


    start_time = time.time()
    train_loader = DataLoader(train_ds, 
                        batch_size=2,  # batch_size=2表示每个batch包含2个样本
                        shuffle=True,  # shuffle=True表示每个epoch随机打乱数据
                        num_workers=1, # num_workers=0表示不使用多线程加载数据
                        drop_last=True # drop_last=True表示如果最后一个batch不满2个样本，则丢弃
                        )
    print(f"采用 drop_last=True, num_workers=1")
    for i, (X, y) in enumerate(train_loader):
        print(f"Batch {i+1}: {X=},{y=}")
    end_time = time.time()
    print(f"数据加载时间：{end_time - start_time:.4f}秒")


    start_time = time.time()
    train_loader = DataLoader(train_ds, 
                        batch_size=2,  # batch_size=2表示每个batch包含2个样本
                        shuffle=True,  # shuffle=True表示每个epoch随机打乱数据
                        num_workers=2, # num_workers=0表示不使用多线程加载数据
                        drop_last=True # drop_last=True表示如果最后一个batch不满2个样本，则丢弃
                        )
    print(f"采用 drop_last=True, num_workers=2")
    for i, (X, y) in enumerate(train_loader):
        print(f"Batch {i+1}: {X=},{y=}")
    end_time = time.time()
    print(f"数据加载时间：{end_time - start_time:.4f}秒")

def compute_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total_examples = 0

    for idx, (features, labels) in enumerate(data_loader):
        with torch.no_grad():
            logits = model(features)
            probs = F.softmax(logits, dim=1)
            predicted_labels = torch.argmax(probs, dim=1)
            correct += (predicted_labels == labels).sum().item()
            total_examples += labels.size(0)
    accuracy = correct / total_examples 
    print("准确率：", accuracy)
    return accuracy
def main():
    model = NeralNetwork(50, 3)
    print(model)
    print("模型参数数量：", count_parameters(model))

    print(model.layers[0].weight)
    # 参数shape 为(30, 50)
    print("参数形状为",  model.layers[0].weight.shape)
    
    torch.manual_seed(123)
    x = torch.randn(1, 50)
    y = model(x)
    print(y)

    # 测试加载时间
    test_data_loader()

    train_loader, test_loader = get_data()
    # 完整训练流程
    torch.manual_seed(123)
    model = NeralNetwork(2, 2)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.5
    )

    num_epochs = 3
    for epoch in range(num_epochs):
        
        model.train() # 需解釋含義
        for batch_idx, (features, labels) in enumerate(train_loader):
            # 前向传播
            logits = model(features)
            # 计算损失
            loss = F.cross_entropy(logits, labels)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            ### LOGGING
            print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
                f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
                f" | Train/Val Loss: {loss:.2f}")

    # 进行预测
    model.eval()
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    with torch.no_grad():
        output = model(torch.tensor(X_train, dtype=torch.float32))    
        print("预测结果：", output)

    torch.set_printoptions(precision=4, sci_mode=False)
    probs =  F.softmax(output, dim=1)
    print("预测概率：", probs)

    print("预测标签：", torch.argmax(probs, dim=1))

    print(compute_accuracy(model, train_loader))

    # save model
    torch.save(model.state_dict(), "model.pth")
    print("模型已保存至 model.pth")
    # load model
    model = NeralNetwork(2, 2)
    model.load_state_dict(torch.load("model.pth"))
    print("模型已加载")


    count_parameters(model)
if __name__ == "__main__":
    # 测试
    main()
