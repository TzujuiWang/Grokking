from model_kit import *
if __name__ == "__main__":
    p = 97
    K = 2
    model_type = "MLP"  # 可以尝试 "MLP", "LSTM", "Transformer"
    optimizer_type = "AdamW"
    num_epochs = 10000
    batch_size = 256
    hidden_dim = 128  # 每个隐藏层的神经元数量
    num_layers = 3     # 设置为3层 MLP
    embed_dim = 64     # 对于 MLP 使用嵌入维度
    learning_rate = 0.001
    weight_decay = 0.01
    alpha = 0.5

    # 根据模型类型设置 input_dim
    input_dim = p + 2  # MLP 的嵌入层 num_embeddings
    input_dim = p + 2  # 对于 LSTM 和 Transformer

    save_path = f"alpha_{alpha}_accuracy_curves_{model_type}.png"
    print(f"Training with alpha={alpha}...")

    # 数据生成
    train_data, train_labels, test_data, test_labels = generate_data(p, K, alpha)

    # 模型、优化器和调度器
    model = select_model(
        model_type,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=p + 2,  # 根据你的任务需求，可能需要调整
        embed_dim=embed_dim,
        num_layers=num_layers,
        dropout=0.1,
        num_heads=4  # 对于 MLP 可以忽略
    )

    optimizer = select_optimizer(optimizer_type, model, lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # 训练模型
    train_losses, test_losses, train_accuracies, test_accuracies = train_model(
        model, optimizer, criterion, train_data, train_labels, test_data, test_labels,
        num_epochs=num_epochs, batch_size=batch_size, scheduler=scheduler, p=p, model_type=model_type
    )

    # 绘制结果
    plot_metrics(train_losses, test_losses, "Loss", f"{model_type} Loss for alpha={alpha}")
    plot_accuracy(train_accuracies, test_accuracies, f"{model_type} Accuracy for alpha={alpha}", save_path)
