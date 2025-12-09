import torch
import numpy as np

from preprocessing import resize_signal, to_recurrence_plot


class ERPDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (self.x[index], self.y[index])


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.data = torch.tensor(
            np.array(
                df.groupby("timestamp")["data_arr"]
                .apply(lambda x: np.stack(x, axis=1))
                .to_list()
            ),
            dtype=torch.float32,
        )
        self.labels = torch.tensor(
            np.array(df.groupby("timestamp")["code"].first().to_list())
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]

        return data, label


class RecurrencePlotDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df.copy()
        self.df.loc[:, "data_arr"] = self.df["data_arr_original"].apply(
            lambda x: resize_signal(x, 28)
        )

        self.data = torch.tensor(
            np.array(
                self.df.groupby("timestamp")["data_arr"]
                .apply(lambda x: to_recurrence_plot(x))
                .to_list()
            ),
            dtype=torch.float32,
        )
        self.labels = torch.tensor(
            np.array(self.df.groupby("timestamp")["code"].first().to_list())
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]

        return data, label
