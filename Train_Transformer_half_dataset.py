from model_kit import *
if __name__ == "__main__":
    p = 97  # 模数
    K = 2  # 输入数字的数量
    model_type = "Transformer"  # 可选 "MLP", "LSTM", "Transformer"
    optimizer_type = "AdamW"
    num_epochs = 10000
    batch_size = 256
    hidden_dim = 128
    learning_rate = 0.001
    weight_decay = 1

    for alpha in [0.3, 0.5, 0.7]:
        print(f"Training with alpha={alpha} on half dataset (using commutative property)...")
        save_path = f"alpha_{alpha}_half_dataset_accuracy_curves_{model_type}.png"

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

        # 训练模型
        train_losses, test_losses, train_accuracies, test_accuracies = train_model(
            model, optimizer, criterion, train_data, train_labels, test_data, test_labels,
            num_epochs=num_epochs, batch_size=batch_size, scheduler=scheduler, p=p
        )

        # 绘制结果
        plot_metrics(train_losses, test_losses, "Loss", f"{model_type} Loss for alpha={alpha} (half dataset)")
        plot_accuracy(
            train_accuracies, test_accuracies, f"{model_type} Accuracy for alpha={alpha} (half dataset)", save_path
        )