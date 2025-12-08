import torch


def get_correct_count(prediction, label):
    return (torch.argmax(prediction, dim=1) == label).sum().item()


def train(
    model,
    loss_function,
    optimizer,
    epochs,
    train_dataloader,
    val_dataloader,
    device,
    save_model="",
):
    history = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}

    model = model.to(device)
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, train_acc = 0, 0

        for data, label in train_dataloader:
            data = data.to(device)
            label = label.to(device)

            # 초기화
            optimizer.zero_grad()

            # prediction 및 loss 계산
            prediction = model(data)
            loss = loss_function(prediction, label)
            train_loss += loss.item()
            train_acc += get_correct_count(prediction, label)

            # Backpropagation
            loss.backward()
            optimizer.step()

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader.dataset)

        # Validation
        model.eval()
        val_loss, val_acc = 0, 0

        with torch.no_grad():
            for data, label in val_dataloader:
                data = data.to(device)
                label = label.to(device)

                prediction = model(data)
                val_loss += loss_function(prediction, label).item()
                val_acc += get_correct_count(prediction, label)

        val_loss /= len(val_dataloader)
        val_acc /= len(val_dataloader.dataset)

        print(
            f"[Epoch {(epoch + 1):02d}] train_acc: {(train_acc * 100):.3f}%, train_loss: {train_loss:.5f}, val_acc: {(val_acc * 100):.3f}%, val_loss: {val_loss:.5f}"
        )

        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

    if save_model:
        torch.save(model.state_dict(), save_model)

    return history
