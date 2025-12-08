import torch


class ERPLSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = torch.nn.LSTM(
            input_size=14, hidden_size=64, num_layers=2, dropout=0.8, batch_first=True
        )
        self.fc = torch.nn.Linear(64, 10)
        self.dropout = torch.nn.Dropout(0.5)

        self.activation = torch.nn.ReLU()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]
        x = self.activation(h_last)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class EEGLSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = torch.nn.LSTM(
            input_size=14, hidden_size=32, num_layers=2, dropout=0.5, batch_first=True
        )
        self.fc = torch.nn.Linear(32, 10)
        self.dropout = torch.nn.Dropout(0.5)

        self.activation = torch.nn.ReLU()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]
        x = self.activation(h_last)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5)
        self.covn2 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5)

        self.fc = torch.nn.Linear(1 * 4 * 4, 10)

        self.dropout = torch.nn.Dropout(0.75)

        self.activation = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)  # (batch_size, 28, 28) -> (batch_size, 1, 28, 28)
        x = self.activation(
            self.conv1(x)
        )  # (batch_size, 1, 28, 28) -> (batch_size, 1, 24, 24)
        x = self.pool(x)  # (batch_size, 1, 24, 24) -> (batch_size, 1, 12, 12)

        x = self.dropout(x)

        x = self.activation(
            self.conv1(x)
        )  # (batch_size, 1, 12, 12) -> (batch_size, 1, 8, 8)
        x = self.pool(x)  # (batch_size, 1, 8, 8) -> (batch_size, 1, 4, 4)

        x = self.dropout(x)

        x = torch.reshape(
            x, (-1, 1 * 4 * 4)
        )  # (batch_size, 1, 4, 4) -> (batch_size, 1 * 4 * 4)
        x = self.fc(x)  # (batch_size, 1 * 4 * 4) -> (batch_size, 10)

        return x
