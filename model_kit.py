import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from itertools import product
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 数据生成函数
# 数据生成函数，支持一半数据集（去除交换律重复样本）
def generate_data(p, K=2, alpha=1.0, use_commutative=False):
    """
    生成数据集，并支持去除交换律重复样本（use_commutative=True）。
    :param p: 模加法的模数
    :param K: 输入数字的数量
    :param alpha: 训练数据占比
    :param use_commutative: 是否利用交换律去重
    """
    total_data = []
    total_labels = []
    seen_pairs = set()  # 用于存储已生成的 (x, y) 或 (y, x)

    for numbers in product(range(p), repeat=K):
        input_tokens = []
        for i, num in enumerate(numbers):
            input_tokens.append(num)
            if i < K - 1:
                input_tokens.append(p)  # 插入加法符号
        input_tokens.append(p + 1)  # 插入等号符号

        if use_commutative:
            # 交换律去重：按升序排列输入对
            sorted_numbers = tuple(sorted(numbers))
            if sorted_numbers in seen_pairs:
                continue  # 如果重复，则跳过
            seen_pairs.add(sorted_numbers)

        total_data.append(tuple(input_tokens))
        total_labels.append(sum(numbers) % p)

    # 打乱并划分训练集和测试集
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
        return LSTMModel(input_dim, embed_dim, hidden_dim, output_dim, num_layers=num_layers, dropout=dropout).to(device)
    elif model_type == "Transformer":
        return TransformerModel(input_dim, hidden_dim, output_dim, num_heads=num_heads, num_layers=num_layers,
                                dropout=dropout).to(device)
    else:
        raise ValueError("Invalid model type. Choose from 'MLP', 'LSTM', or 'Transformer'.")


# 优化器选择
def select_optimizer(optimizer_type, model, lr, weight_decay=0.0):
    if optimizer_type == "AdamW":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
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


def train_model_with_l2_norm(
    model, optimizer, criterion, train_data, train_labels, test_data, test_labels,
    num_epochs=100, batch_size=32, save_every=10000, scheduler=None, p=None, model_type=None
):
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    weight_norms = []  # 用于记录权重 L2 范数

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
            epoch_loss += loss.item() * batch_inputs.size(0)
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

        # 记录权重范数
        l2_norm = sum(param.norm(2).item() for param in model.parameters())
        weight_norms.append(l2_norm)

        # 调用学习率调度器
        if scheduler:
            scheduler.step()

        # 打印每 10 个 epoch 的信息
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, "
                  f"Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.4f}, "
                  f"L2 Norm: {l2_norm:.4f}")

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
                'weight_norms': weight_norms,
            }, f'checkpoint_epoch_{epoch + 1}.pth')

    return train_losses, test_losses, train_accuracies, test_accuracies, weight_norms


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
def plot_metrics(train_metrics, test_metrics, metric_name, title, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(train_metrics, label=f"Train {metric_name}")
    plt.plot(test_metrics, label=f"Test {metric_name}", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_weight_norms(train_accuracies, test_accuracies, weight_norms, title="Weight Norm and Accuracy Curve", save_path=None):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # 左边的 Y 轴：Accuracy
    ax1.plot(train_accuracies, label="Train Accuracy", color="tab:blue")
    ax1.plot(test_accuracies, label="Test Accuracy", linestyle="--", color="tab:orange")
    ax1.set_xlabel("Epochs (Log Scale)")
    ax1.set_ylabel("Accuracy", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xscale("log")  # 设置横坐标为对数刻度

    # 添加网格
    ax1.grid()

    # 右边的 Y 轴：L2 Norm
    ax2 = ax1.twinx()
    ax2.plot(weight_norms, label="Weight L2 Norm", color="tab:green")
    ax2.set_ylabel("L2 Norm", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    # 添加标题
    fig.suptitle(title)

    # 添加图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    # 保存图像
    if save_path:
        plt.savefig(save_path)
    plt.show()



if __name__ == "__main__":
    p = 97  # 模数
    K = 2  # 输入数字的数量
    model_type = "Transformer"  # 可选 "MLP", "LSTM", "Transformer"
    optimizer_type = "AdamW"
    num_epochs = 1500
    batch_size = 256
    hidden_dim = 128
    learning_rate = 0.001
    weight_decay = 0.5

    for alpha in [0.5]:
        print(f"Training with alpha={alpha} on half dataset (using commutative property)...")
        save_path_accuracy = f"alpha_{alpha}_accuracy_curves_{model_type}.png"
        save_path_weight_norm = f"alpha_{alpha}_weight_norm_{model_type}.png"
        save_path_loss = f"alpha_{alpha}_loss_{model_type}.png"

        # 数据生成，启用交换律去重
        train_data, train_labels, test_data, test_labels = generate_data(
            p, K, alpha, use_commutative=True
        )

        # 模型、优化器和调度器
        model = select_model(
            model_type, input_dim=p + 2, hidden_dim=hidden_dim, output_dim=p + 2, num_heads=4, num_layers=2
        )
        optimizer = select_optimizer(optimizer_type, model, lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.1)
        criterion = nn.CrossEntropyLoss()

        # 训练模型并记录权重范数
        train_losses, test_losses, train_accuracies, test_accuracies, weight_norms = train_model_with_l2_norm(
            model, optimizer, criterion, train_data, train_labels, test_data, test_labels,
            num_epochs=num_epochs, batch_size=batch_size, scheduler=scheduler, p=p
        )

        # 绘制权重范数曲线
        plot_weight_norms(train_accuracies, test_accuracies, weight_norms,
                          f"{model_type} Weight Norm for alpha={alpha}", save_path_weight_norm)

        # 绘制准确率曲线
        plot_metrics(train_losses, test_losses, "Loss",
                     f"{model_type} Loss for alpha={alpha}", save_path_loss)