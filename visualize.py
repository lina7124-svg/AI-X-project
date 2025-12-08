import os
import matplotlib.pyplot as plt


def save_figure(figure, path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    figure.savefig(path)


def visualize_event(data, save_file=""):
    channels = data["channel"]

    fig, axes = plt.subplots(len(channels), 1, figsize=(10, 2 * len(channels)))

    fig.set_layout_engine("tight")
    fig.suptitle(f"EEG (code: {data["code"].unique().item()})")

    for channel, ax in zip(channels, axes):
        eeg = data[data["channel"] == channel]["data_arr"].item()
        ax.plot(eeg)
        ax.set_xticks([])
        ax.set_ylabel(channel)
        ax.yaxis.set_ticks_position("right")

    plt.show()
    if save_file:
        save_figure(fig, save_file)

    plt.close(fig)


def visualize_class_count(df, save_file=""):
    fig, ax = plt.subplots(figsize=(10, 5))

    fig.set_layout_engine("tight")
    fig.suptitle("Data Distribution")

    bar = ax.bar(
        range(0, 10),
        df.groupby("timestamp")["code"].first().value_counts().sort_index(),
    )
    ax.set_xticks(range(0, 10))
    ax.bar_label(bar)

    plt.show()
    if save_file:
        save_figure(fig, save_file)

    plt.close(fig)


def visualize_erp(dataset, channels, idx=0, save_file=""):
    sig = dataset[idx][0]
    label = dataset[idx][1]

    fig = plt.figure(figsize=(12, 6))
    plt.title(f"ERP Multi-channel Sample #{idx}, code={label}")

    for ch_idx, ch in enumerate(channels):
        plt.plot(sig[:, ch_idx], label=ch)

    plt.legend()
    plt.grid()

    plt.show()
    if save_file:
        save_figure(fig, save_file)
    plt.close(fig)


def visualize_recurrence_plot(data, label, save_file=""):
    fig, ax = plt.subplots(figsize=(5, 5))

    fig.set_layout_engine("tight")
    fig.suptitle(f"Recurrence Plot (code: {label})")

    ax.imshow(data, cmap="gray")

    plt.show()
    if save_file:
        save_figure(fig, save_file)
    plt.close(fig)


def visualize_train_history(history, model_name, save_loss_file="", save_acc_file=""):
    ticks = range(10, len(history["train_loss"]) + 1, 10)

    # Loss
    fig, ax = plt.subplots(figsize=(5, 5))

    fig.set_layout_engine("tight")
    fig.suptitle(
        f"Loss (model: {model_name})",
        fontsize=12,
        fontweight="bold",
        y=0.95,
    )

    ax.plot(history["train_loss"], label="Train Loss", color="red")
    ax.plot(history["val_loss"], label="Validation Loss", color="blue")

    ax.set_xticks(ticks)
    ax.set_xlabel("Epoch", fontsize=10, fontweight="semibold")
    ax.legend()

    plt.show()
    if save_loss_file:
        save_figure(fig, save_loss_file)
    plt.close(fig)

    # Accuracy
    fig, ax = plt.subplots(figsize=(5, 5))

    fig.set_layout_engine("tight")
    fig.suptitle(
        f"Accuracy (model: {model_name})",
        fontsize=12,
        fontweight="bold",
        y=0.95,
    )

    ax.plot(history["train_acc"], label="Train Accuracy", color="red")
    ax.plot(history["val_acc"], label="Validation Accuracy", color="blue")

    ax.set_xticks(ticks)
    ax.set_xlabel("Epoch", fontsize=10, fontweight="semibold")
    ax.legend()

    plt.show()
    if save_acc_file:
        save_figure(fig, save_acc_file)
    plt.close(fig)


if __name__ == "__main__":
    from preprocessing import df_ep

    sample_timestamp = df_ep.iloc[0]["timestamp"]
    visualize_event(
        df_ep[df_ep["timestamp"] == sample_timestamp],
        save_file=f"./figures/example-EEG.png",
    )

    visualize_class_count(df_ep, save_file="./figures/class-count.png")
