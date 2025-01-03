from model_kit import *
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