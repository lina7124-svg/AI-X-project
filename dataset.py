# 데이터 불러오기
# MindBigData Dataset https://www.kaggle.com/datasets/vijayveersingh/1-2m-brain-signal-data

import numpy as np
import pandas as pd
import torch


def read_eeg_file(file_path):
    return pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=["id", "event", "device", "channel", "code", "size", "data"],
        index_col="id",
        converters={
            "data": lambda data: np.array(
                [np.float32(amplitude) for amplitude in data.split(",")]
            )
        },
    )


df_EP = read_eeg_file("./data/MindBigData-EP-v1.0/EP1.01.txt")
df_IN = read_eeg_file("./data/MindBigData-IN-v1.06/IN.txt")
df_MU = read_eeg_file("./data/MindBigData-MU-v1.0/MU.txt")
df_MW = read_eeg_file("./data/MindBigData-MW-v1.0/MW.txt")
df = pd.concat([df_EP, df_IN, df_MU, df_MW])


def normalize(df):
    devices = df["device"].unique()
    for device in devices:
        df_device = df[df["device"] == device]
        all_amplitude = df_device["data"].explode().to_list()
        mean = np.mean(all_amplitude)
        std = np.std(all_amplitude)
        df.loc[df["device"] == device, "data"] = (df_device["data"] - mean) / std
    return df


def add_padding(df):
    max_size = max(df["size"].unique())
    df.loc[:, "data"] = df["data"].apply(
        lambda eeg: np.pad(eeg, (0, max_size), mode="constant", constant_values=0)[
            :max_size
        ]
    )
    return df, max_size


class EEGDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        channel,
        normalization=True,
        padding=True,
        transform=torch.tensor,
        target_transform=lambda label: torch.zeros(10, dtype=torch.float).scatter_(
            0, torch.tensor(label), value=1
        ),
    ):
        self.channel = channel

        self.df = df_device[
            (df_device["channel"] == channel) & (df_device["code"] != -1)
        ]
        if normalization:
            self.df = normalize(self.df)
        if padding:
            self.df, self.eeg_length = add_padding(self.df)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        data = self.df.iloc[index]["data"]
        if self.transform:
            data = self.transform(data)

        label = self.df.iloc[index]["code"]
        if self.target_transform:
            label = self.target_transform(label)

        return data, label


# 데이터를 그래프로 표현

import matplotlib.pyplot as plt


def visualize_single_channel(data, label):
    fig, ax = plt.subplots(figsize=(10, 2))

    fig.set_layout_engine("tight")
    fig.suptitle(f"EEG (code: {label})")

    ax.plot(data)
    ax.set_xticks([])

    plt.show()

    return fig, ax


def visualize_event(data):
    channels = data["channel"]

    fig, axes = plt.subplots(len(channels), 1, figsize=(10, 2 * len(channels)))
    axes = [axes] if len(channels) == 1 else axes  # axes는 항상 list

    fig.set_layout_engine("tight")
    fig.suptitle(
        f"EEG (code: {data["code"].unique().item()}, device: {data["device"].unique().item()})"
    )

    for channel, ax in zip(channels, axes):
        eeg = data[data["channel"] == channel]["data"].item()
        ax.plot(eeg)
        ax.set_xticks([])
        ax.set_ylabel(channel)
        ax.yaxis.set_ticks_position("right")

    plt.show()

    return fig, axes


if __name__ == "__main__":
    # 각 기기별 예시 데이터
    for df_device in [df_EP, df_IN, df_MU, df_MW]:
        event = df_device.iloc[0]["event"]
        device = df_device.iloc[0]["device"]
        fig, _ = visualize_event(df_device[df_device["event"] == event])
        fig.savefig(f"./figures/eeg-{device}.png")
        print(f"{device} 총 데이터 수: {df_device.shape[0]}")

    # code == -1인 데이터 제외
    df = df[df["code"] != -1]

    # code별 eeg 데이터 수
    channels = ["AF3", "AF4", "FP1"]
    x = np.arange(10)
    fig, ax = plt.subplots(figsize=(20, 5))
    fig.set_layout_engine("tight")
    for i, channel in enumerate(channels):
        bar = ax.bar(
            x + 0.25 * (i - 1),
            df[df["channel"] == channel]["code"].value_counts().sort_index(),
            width=0.25,
            label=channel,
        )
        ax.bar_label(bar, padding=3, fontsize=7)
    ax.legend(loc="upper left", ncols=3)
    ax.set_xticks(x)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

    plt.show()
    fig.savefig(f"./figures/class-count.png")
