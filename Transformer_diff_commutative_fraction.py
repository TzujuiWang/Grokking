from model_kit import *
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from itertools import product
from sklearn.model_selection import train_test_split


def generate_data(p, K=2, alpha=1.0, commutative_fraction=0.5):
    """
    生成数据集，并通过调整训练集的样本来源来控制交换律分数。
    :param p: 模加法的模数
    :param K: 输入数字的数量
    :param alpha: 训练数据占比
    :param commutative_fraction: 训练集中的交换律分数（0~1）
    """
    total_data = []
    total_labels = []
    seen_pairs = set()

    # Step 1: 生成所有可能的 K 元组组合并去掉重复项 (只保留 x, y 或 y, x 中的一个)
    for numbers in product(range(p), repeat=K):
        sorted_numbers = tuple(sorted(numbers))  # 通过排序确保 x, y 与 y, x 视为相同
        if sorted_numbers not in seen_pairs:
            seen_pairs.add(sorted_numbers)

            # 构造输入序列 (x, y, +, =)
            input_tokens = []
            for i, num in enumerate(numbers):
                input_tokens.append(num)
                if i < K - 1:
                    input_tokens.append(p)  # 插入加法符号
            input_tokens.append(p + 1)  # 插入等号符号

            total_data.append(tuple(input_tokens))
            total_labels.append(sum(numbers) % p)

    # Step 2: 划分训练集 (T) 和测试集 (V)
    train_data, test_data, train_labels, test_labels = train_test_split(
        total_data, total_labels, train_size=alpha, shuffle=True, random_state=42
    )

    # Step 3: 调整训练集的 commutative fraction
    # 计算训练集中需要的非交换律样本数量和交换律样本数量
    initial_train_size = len(train_data)
    target_commutative_count = int(initial_train_size * commutative_fraction)
    target_non_commutative_count = initial_train_size - target_commutative_count

    # 从当前训练集中抽取 (1 - commutative_fraction) 的非交换律样本
    updated_train_data = train_data[:target_non_commutative_count]
    updated_train_labels = train_labels[:target_non_commutative_count]

    # 从测试集中选取 commutative_fraction 的样本，反转其 x 和 y
    selected_test_data = test_data[:target_commutative_count]
    selected_test_labels = test_labels[:target_commutative_count]

    commutative_samples = []
    for sample in selected_test_data:
        reversed_sample = (sample[2], sample[1], sample[0], sample[3])  # 交换 x 和 y
        commutative_samples.append(reversed_sample)

    # 将交换后的样本添加到训练集中
    updated_train_data.extend(commutative_samples)
    updated_train_labels.extend(selected_test_labels)

    return updated_train_data, updated_train_labels, test_data, test_labels


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

def plot_9_subplots(data_dict, main_title, metric_name, save_path=None):
    """
    绘制包含 9 个子图的图像，每个子图显示 Train/Test Accuracy 或 Train/Test Loss。
    :param data_dict: 包含多个数据的字典，键是子图标题，值是 (epochs, train_metric, test_metric) 的元组。
    :param main_title: 图像的主标题。
    :param metric_name: 要绘制的度量名称（"Accuracy" 或 "Loss"）。
    :param save_path: 如果提供，将保存图像到指定路径。
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))  # 创建 3x3 网格子图
    fig.suptitle(main_title, fontsize=20)  # 设置主标题

    for idx, (title, (epochs, train_metric, test_metric)) in enumerate(data_dict.items()):
        row, col = divmod(idx, 3)  # 确定子图的位置
        ax = axes[row, col]

        # 绘制训练和测试曲线
        ax.plot(epochs, train_metric, label=f"Train {metric_name}", color="blue")
        ax.plot(epochs, test_metric, label=f"Test {metric_name}", color="orange", linestyle="--")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epochs")
        ax.set_ylabel(metric_name)
        ax.set_xscale("log")  # 对数横轴
        ax.legend()
        ax.grid()

    # 删除多余的子图
    for idx in range(len(data_dict), 9):
        row, col = divmod(idx, 3)
        fig.delaxes(axes[row, col])

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整布局
    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    # 检查 commutative fraction
    # def check_commutative_fraction(train_data, test_data):
    #     """
    #     检查训练集中的 commutative fraction，即训练集中的 (x, y) 是否出现在测试集中作为 (y, x)。
    #     :param train_data: 训练数据集
    #     :param test_data: 测试数据集
    #     :return: commutative fraction
    #     """
    #     commutative_pairs = 0
    #     for sample in train_data:
    #         # 提取训练集的样本格式 (y, +, x, =)
    #         reversed_sample = (sample[2], sample[1], sample[0], sample[3])  # 反转为 (x, +, y, =)
    #         if reversed_sample in test_data:
    #             commutative_pairs += 1  # 如果反转样本存在于测试集中，计数加一
    #
    #     return commutative_pairs / len(train_data)
    #
    #
    # train_data, train_labels, test_data, test_labels = generate_data(
    #     p=97, K=2, alpha=0.5, commutative_fraction=0.5
    # )
    # print(len(train_data))
    # print(len(test_data))
    #
    # # 检查训练集的 commutative fraction
    # commutative_fraction = check_commutative_fraction(train_data, test_data)
    # print(f"实际交换律分数: {commutative_fraction:.2f}")
    p = 97  # 模数
    K = 2  # 输入数字的数量
    model_type = "Transformer"  # 可选 "MLP", "LSTM", "Transformer"
    optimizer_type = "AdamW"
    num_epochs = 2000
    batch_size = 256
    hidden_dim = 128
    learning_rate = 0.001
    weight_decay = 0.5

    # 不同 commutative_fraction 设置
    commutative_fractions = [round(i * 0.1, 1) for i in range(1, 10)]  # [0.1, 0.2, ..., 0.9]
    accuracy_dict = {}
    loss_dict = {}

    for commutative_fraction in commutative_fractions:
        title = f"Comm={commutative_fraction}"
        print(f"Training with {title}...")

        # 数据生成，启用交换律去重
        train_data, train_labels, test_data, test_labels = generate_data(
            p, K, alpha=0.5, commutative_fraction=commutative_fraction
        )

        # 模型、优化器和调度器
        model = select_model(
            model_type, input_dim=p + 2, hidden_dim=hidden_dim, output_dim=p + 2, num_heads=4, num_layers=2
        )
        optimizer = select_optimizer(optimizer_type, model, lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.1)
        criterion = nn.CrossEntropyLoss()

        # 训练模型
        train_losses, test_losses, train_accuracies, test_accuracies = train_model(
            model, optimizer, criterion, train_data, train_labels, test_data, test_labels,
            num_epochs=num_epochs, batch_size=batch_size, scheduler=scheduler, p=p
        )

        # 将 Accuracy 和 Loss 分别存储到字典中
        epochs = list(range(1, num_epochs + 1))
        accuracy_dict[title] = (epochs, train_accuracies, test_accuracies)
        loss_dict[title] = (epochs, train_losses, test_losses)

    # 绘制 9 子图的大图 (Accuracy)
    plot_9_subplots(
        accuracy_dict,
        main_title=f"{model_type} Accuracy Curves for Different Commutative Fractions",
        metric_name="Accuracy",
        save_path="4-2/accuracy_9_subplots.png"
    )

    # 绘制 9 子图的大图 (Loss)
    plot_9_subplots(
        loss_dict,
        main_title=f"{model_type} Loss Curves for Different Commutative Fractions",
        metric_name="Loss",
        save_path="4-2/loss_9_subplots.png"
    )

