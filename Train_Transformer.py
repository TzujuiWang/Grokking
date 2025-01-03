from model_kit import *
# 训练不同alpha下的Transformer
if __name__ == "__main__":
    p = 97
    K = 2
    model_type = "Transformer"
    optimizer_type = "AdamW"
    num_epochs = 10000
    batch_size = 256
    hidden_dim = 128
    learning_rate = 0.001
    weight_decay = 1

    for alpha in [0.3, 0.5, 0.7]:
        save_path = f"alpha_{alpha}_accuracy_curves_{model_type}.png"
        print(f"Training with alpha={alpha}...")

        # 数据生成
        train_data, train_labels, test_data, test_labels = generate_data(p, K, alpha)

        # 模型、优化器和调度器
        model = select_model(model_type, input_dim=p+2, hidden_dim=hidden_dim, output_dim=p+2, num_heads=4, num_layers=2)
        optimizer = select_optimizer(optimizer_type, model, lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)
        criterion = nn.CrossEntropyLoss()

        # 训练模型
        train_losses, test_losses, train_accuracies, test_accuracies = train_model(
            model, optimizer, criterion, train_data, train_labels, test_data, test_labels,
            num_epochs=num_epochs, batch_size=batch_size, scheduler=scheduler, p = p
        )

        # 绘制结果
        plot_metrics(train_losses, test_losses, "Loss", f"{model_type} Loss for alpha={alpha}")
        plot_accuracy(train_accuracies, test_accuracies, f"{model_type} Accuracy for alpha={alpha}", save_path)
