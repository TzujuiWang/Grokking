import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from itertools import product
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pickle
import os

# 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 数据生成函数
def generate_data(p, K=2, alpha=1.0):
    total_data = []
    total_labels = []
    for numbers in product(range(p), repeat=K):
        input_tokens = []
        for i, num in enumerate(numbers):
            input_tokens.append(num)
            if i < K - 1:
                input_tokens.append(p)
        input_tokens.append(p + 1)
        total_data.append(tuple(input_tokens))
        total_labels.append(sum(numbers) % p)

    # 转换为列表以便打乱
    total_data = list(total_data)
    total_labels = list(total_labels)

    # 随机打乱数据
    train_data, test_data, train_labels, test_labels = train_test_split(
        total_data, total_labels, train_size=alpha, shuffle=True, random_state=42
    )

    return train_data, train_labels, test_data, test_labels


# 数据预处理
def preprocess_data(data, labels, p, model_type=None):
    inputs = torch.tensor(data, dtype=torch.long).to(device)
    assert inputs.min() >= 0 and inputs.max() < p + 2, f"Input values must be in range [0, {p + 1}] but got [{inputs.min()}, {inputs.max()}]"
    inputs = inputs.view(-1, len(data[0]))  # [batch_size, seq_len]

    targets = torch.tensor(labels, dtype=torch.long).to(device)
    return inputs, targets


# MLP 模型
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, embed_dim=64, dropout_rate=0.0):
        super(MLPModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=embed_dim)
        # The total input dimension after flattening the embeddings
        total_input_dim = 4 * embed_dim  # 4 tokens each with embed_dim dimensions

        layers = []
        current_dim = total_input_dim

        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # x: [batch_size, seq_len]
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        x = x.view(x.size(0), -1)  # [batch_size, seq_len * embed_dim]
        return self.model(x)  # [batch_size, output_dim]


# LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim]
        return self.fc_out(lstm_out[:, -1, :])  # 取最后一个时间步的输出


# Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)  # 输出 [batch_size, seq_len, hidden_dim]
        x = self.transformer(x)  # TransformerEncoder 输入 [batch_size, seq_len, hidden_dim]
        return self.fc_out(x[:, -1, :])  # 取最后一个时间步的输出


# 模型选择
def select_model(model_type, input_dim, hidden_dim, output_dim, embed_dim=64, num_layers=2, dropout=0.0, num_heads=4):
    if model_type == "MLP":
        # 对于 MLP，input_dim 是嵌入层的 num_embeddings
        return MLPModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                        output_dim=output_dim, embed_dim=embed_dim, dropout_rate=dropout).to(device)
    elif model_type == "LSTM":
        return LSTMModel(input_dim, embed_dim, hidden_dim, output_dim, num_layers=num_layers, dropout=dropout).to(
            device)
    elif model_type == "Transformer":
        return TransformerModel(input_dim, hidden_dim, output_dim, num_heads=num_heads, num_layers=num_layers,
                                dropout=dropout).to(device)
    else:
        raise ValueError("Invalid model type. Choose from 'MLP', 'LSTM', or 'Transformer'.")


# 优化器选择
def select_optimizer(optimizer_type, model, lr, weight_decay=0.0):
    if optimizer_type == "AdamW":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "SGD":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("Unsupported optimizer type.")


def train_model(model, optimizer, criterion, train_data, train_labels, test_data, test_labels, num_epochs=100,
                batch_size=32, save_every=10000, scheduler=None, p=None, model_type=None):
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    # 预处理数据
    train_inputs, train_targets = preprocess_data(train_data, train_labels, p, model_type=model_type)
    test_inputs, test_targets = preprocess_data(test_data, test_labels, p, model_type=model_type)

    # 创建数据加载器
    train_dataset = TensorDataset(train_inputs, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        # 按批次训练
        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

            # 累加损失和准确率
            epoch_loss += loss.item() * batch_inputs.size(0)  # 按批次累计总损失
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == batch_targets).sum().item()
            total += batch_targets.size(0)

        # 记录每个 epoch 的平均训练损失和准确率
        train_losses.append(epoch_loss / total)
        train_accuracies.append(correct / total)

        # 测试模型
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_inputs)
            test_loss = criterion(test_outputs, test_targets)
            predicted_test = torch.argmax(test_outputs, dim=1)
            test_accuracy = (predicted_test == test_targets).sum().item() / len(test_targets)

            # 记录测试损失和准确率
            test_losses.append(test_loss.item())
            test_accuracies.append(test_accuracy)

        # 调用学习率调度器
        if scheduler:
            scheduler.step()

        # 打印每 10 个 epoch 的信息
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, "
                  f"Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.4f}")

        # 保存模型
        if (epoch + 1) % save_every == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_losses': train_losses,
                'test_losses': test_losses,
                'train_accuracies': train_accuracies,
                'test_accuracies': test_accuracies,
            }, f'checkpoint_epoch_{epoch + 1}.pth')

    return train_losses, test_losses, train_accuracies, test_accuracies


def plot_accuracy(train_accuracies, test_accuracies, title="Accuracy Curve", save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(test_accuracies, label="Test Accuracy", linestyle="--")
    plt.xlabel("Optimization Steps (Log Scale)")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.xscale("log")  # 设置横坐标为对数刻度
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# 绘制图像
def plot_metrics(train_metrics, test_metrics, metric_name, title):
    plt.figure(figsize=(10, 5))
    plt.plot(train_metrics, label=f"Train {metric_name}")
    plt.plot(test_metrics, label=f"Test {metric_name}", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_optimizer_comparison(results, title="Optimizer Comparison", save_path=None):
    # 假设 results 是一个字典：
    # {
    #   "Optimizer1": {"alpha": [...], "accuracy": [...]},
    #   "Optimizer2": {"alpha": [...], "accuracy": [...]},
    #    ...
    # }

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # 创建 2x3 网格子图
    axes = axes.flatten()  # 将axes数组拉平为一维，方便迭代

    for ax, (name, result) in zip(axes, results.items()):
        ax.plot(result["alpha"], result["accuracy"], 'o-', label=name)
        ax.set_title(name)
        ax.set_xlabel("Training data fraction (alpha)")
        ax.set_ylabel("Best validation accuracy")
        ax.grid(True)
        # 设置X轴和Y轴范围都为[0,1]
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()

    plt.suptitle(title, y=1.02)
    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    p = 97
    K = 2
    model_type = "Transformer"  # 可以根据需要修改为 "MLP", "LSTM", 或 "Transformer"
    num_epochs = 10000
    # num_epochs = 10
    batch_size = 256
    hidden_dim = 128
    alpha_values = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]
    # alpha_values = [0.2]

    # 定义一系列实验配置，每个配置是一个字典：
    # 包含 "name" (显示在图例中), "optimizer_type", "lr", "weight_decay", "dropout", "momentum", "nesterov" 等参数
    # 未指定的参数可以使用默认值，例如 dropout=0.0, weight_decay=0.0等。
    experiments = [
        {
            "name": "Minibatch Adam",
            "optimizer_type": "AdamW",  # 使用AdamW，但设置weight_decay=0，相当于Adam
            "lr": 0.001,
            "weight_decay": 0.0,
            "dropout": 0.0,
            "momentum": 0.0,
            "nesterov": False
        },
        {
            "name": "AdamW, weight decay 0.05",
            "optimizer_type": "AdamW",
            "lr": 0.001,
            "weight_decay": 0.05,
            "dropout": 0.0,
            "momentum": 0.0,
            "nesterov": False
        },
        {
            "name": "AdamW, weight decay 1",
            "optimizer_type": "AdamW",
            "lr": 0.001,
            "weight_decay": 1.0,
            "dropout": 0.0,
            "momentum": 0.0,
            "nesterov": False
        },
        {
            "name": "Dropout 0.1, Adam",
            "optimizer_type": "AdamW",  # 无weight decay则接近Adam
            "lr": 0.001,
            "weight_decay": 0.0,
            "dropout": 0.1,
            "momentum": 0.0,
            "nesterov": False
        },
        {
            "name": "Adam, 0.3x learning rate",
            "optimizer_type": "AdamW",
            "lr": 0.0003,
            "weight_decay": 0.0,
            "dropout": 0.0,
            "momentum": 0.0,
            "nesterov": False
        },
        {
            "name": "SGD with Nesterov, momentum=0.99",
            "optimizer_type": "SGD",
            "lr": 0.001,
            "weight_decay": 0.0,
            "dropout": 0.0,
            "momentum": 0.99,
            "nesterov": True
        }
    ]

    # 存放结果，键为实验名称，值为在不同alpha下的最佳验证集准确率列表
    if os.path.exists('partial_results.pkl'):
        with open('partial_results.pkl', 'rb') as f:
            results = pickle.load(f)
    else:
        results = {}

    save_path = '3-3/optimizer_comparison.png'

    for exp in experiments:
        exp_name = exp["name"]
        if exp_name in results:
            # 已有此实验结果，跳过
            print(f"Skipping {exp_name}, already completed.")
            continue

        print(f"Running experiment: {exp_name}")
        best_accuracies = []

        for alpha in alpha_values:
            print(f"Training with alpha={alpha} for {exp_name}...")

            # 生成数据
            train_data, train_labels, test_data, test_labels = generate_data(p, K, alpha)

            # 构建模型, 注意传递 dropout 参数
            # 对于 Transformer，你可以传递 num_layers、num_heads 等参数
            model = select_model(
                model_type,
                input_dim=p + 2,
                hidden_dim=hidden_dim,
                output_dim=p + 2,
                embed_dim=64,  # 可根据需要调整embed_dim
                num_layers=2,  # Transformer层数可根据需要调整
                dropout=exp["dropout"],
                num_heads=4
            )

            # 根据 optimizer_type 建立优化器, 对于SGD可以考虑额外操作处理 nesterov、momentum
            # 因为 select_optimizer 仅支持基础操作，需要修改 select_optimizer 函数或在此动态处理。
            # 假设select_optimizer不支持nesterov和momentum参数，可以自己在此处处理:
            if exp["optimizer_type"] == "SGD":
                # 自行构建SGD优化器
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=exp["lr"],
                    momentum=exp["momentum"],
                    weight_decay=exp["weight_decay"],
                    nesterov=exp["nesterov"]
                )
            else:
                # 使用select_optimizer
                optimizer = select_optimizer(exp["optimizer_type"], model, lr=exp["lr"],
                                             weight_decay=exp["weight_decay"])

            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)
            criterion = nn.CrossEntropyLoss()

            # 训练模型
            train_losses, test_losses, train_accuracies, test_accuracies = train_model(
                model, optimizer, criterion, train_data, train_labels, test_data, test_labels,
                num_epochs=num_epochs, batch_size=batch_size, scheduler=scheduler, p=p, model_type=model_type
            )

            # 找到该alpha下的最佳验证集准确率
            best_val_acc = max(test_accuracies)
            best_accuracies.append(best_val_acc)

        results[exp_name] = {
            "alpha": alpha_values,
            "accuracy": best_accuracies
        }
        # 每个实验结束后立即保存一次结果
        with open('partial_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        print(f"Results after {exp_name} saved to partial_results.pkl")

    plot_optimizer_comparison(results, save_path=save_path)