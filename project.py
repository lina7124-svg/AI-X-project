import os, random
import torch
import numpy as np


# 재현을 위해 seed 고정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed()


###### 데이터셋 준비 및 확인 ######


from preprocessing import (
    channels,
    X_ERP_train,
    Y_ERP_train,
    X_ERP_val,
    Y_ERP_val,
    df_train,
    df_val,
)
from dataset import ERPDataset, EEGDataset, RecurrencePlotDataset
from visualize import visualize_erp, visualize_recurrence_plot

batch_size = 16

# ERP
erp_train_dataset = ERPDataset(X_ERP_train, Y_ERP_train)
erp_val_dataset = ERPDataset(X_ERP_val, Y_ERP_val)
erp_train_dataloader = torch.utils.data.DataLoader(
    erp_train_dataset, batch_size=batch_size, shuffle=True
)
erp_val_dataloader = torch.utils.data.DataLoader(
    erp_val_dataset, batch_size=batch_size, shuffle=True
)
## 예시 및 크기 확인
visualize_erp(erp_train_dataset, channels, idx=0, save_file="./figures/example-ERP.png")
print(f"ERP Train 데이터셋 크기: {len(erp_train_dataset)}")
print(f"ERP Validation 데이터셋 크기: {len(erp_val_dataset)}")

# EEG
eeg_train_dataset = EEGDataset(df_train)
eeg_val_dataset = EEGDataset(df_val)
train_eeg_dataloader = torch.utils.data.DataLoader(
    eeg_train_dataset, batch_size=batch_size, shuffle=True
)
val_eeg_dataloader = torch.utils.data.DataLoader(
    eeg_val_dataset, batch_size=batch_size, shuffle=True
)
## 크기 확인
print(f"EEG Train 데이터셋 크기: {len(eeg_train_dataset)}")
print(f"EEG Validation 데이터셋 크기: {len(eeg_val_dataset)}")

# CNN with Recurrence Plot
rp_train_dataset = RecurrencePlotDataset(df_train)
rp_val_dataset = RecurrencePlotDataset(df_val)
train_rp_dataloader = torch.utils.data.DataLoader(
    rp_train_dataset, batch_size=batch_size, shuffle=True
)
val_rp_dataloader = torch.utils.data.DataLoader(
    rp_val_dataset, batch_size=batch_size, shuffle=True
)
## 예시 및 크기 확인
visualize_recurrence_plot(
    rp_train_dataset[15][0],
    rp_train_dataset[15][1],
    save_file="./figures/example-RP.png",
)
print(f"CNN Train 데이터셋 크기: {len(rp_train_dataset)}")
print(f"CNN Validation 데이터셋 크기: {len(rp_val_dataset)}")


###### 모델 훈련 ######


from models import ERPLSTM, EEGLSTM, CNN
from train import train
from visualize import visualize_train_history


# GPU 등 있으면 활용
device = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else torch.device("cpu")
)
print(f"Device: {device}")

# 모델 저장할 폴더 생성
os.makedirs("./models", exist_ok=True)

# ERP-LSTM
model_erp_lstm = ERPLSTM()
loss_function_erp_lstm = torch.nn.CrossEntropyLoss()
optimizer_erp_lstm = torch.optim.Adam(
    model_erp_lstm.parameters(), lr=0.000007, weight_decay=1e-5
)
print("----- 모델 훈련: ERP-LSTM -----")
train_history_erp_lstm = train(
    model=model_erp_lstm,
    loss_function=loss_function_erp_lstm,
    optimizer=optimizer_erp_lstm,
    epochs=50,
    train_dataloader=erp_train_dataloader,
    val_dataloader=erp_val_dataloader,
    device=device,
    save_model=f"./models/ERP-LSTM.pth",
)
print()
visualize_train_history(
    train_history_erp_lstm,
    "ERP-LSTM",
    f"./figures/models/ERP-LSTM-loss.png",
    f"./figures/models/ERP-LSTM-accuracy.png",
)

# EEG-LSTM
model_eeg_lstm = EEGLSTM()
loss_function_eeg_lstm = torch.nn.CrossEntropyLoss()
optimizer_eeg_lstm = torch.optim.Adam(
    model_eeg_lstm.parameters(), lr=1e-5, weight_decay=1e-6
)
print("----- 모델 훈련: EEG-LSTM -----")
train_history_eeg_lstm = train(
    model=model_eeg_lstm,
    loss_function=loss_function_eeg_lstm,
    optimizer=optimizer_eeg_lstm,
    epochs=50,
    train_dataloader=train_eeg_dataloader,
    val_dataloader=val_eeg_dataloader,
    device=device,
    save_model=f"./models/EEG-LSTM.pth",
)
print()
visualize_train_history(
    train_history_eeg_lstm,
    "EEG-LSTM",
    f"./figures/models/EEG-LSTM-loss.png",
    f"./figures/models/EEG-LSTM-accuracy.png",
)

# CNN with Recurrence Plot
model_cnn = CNN()
loss_function_cnn = torch.nn.CrossEntropyLoss()
optimizer_cnn = torch.optim.Adam(model_cnn.parameters(), lr=1e-6, weight_decay=1e-6)
print("----- 모델 훈련: CNN -----")
train_history_cnn = train(
    model=model_cnn,
    loss_function=loss_function_cnn,
    optimizer=optimizer_cnn,
    epochs=50,
    train_dataloader=train_rp_dataloader,
    val_dataloader=val_rp_dataloader,
    device=device,
    save_model=f"./models/CNN.pth",
)
print()
visualize_train_history(
    train_history_cnn,
    "CNN",
    f"./figures/models/CNN-loss.png",
    f"./figures/models/CNN-accuracy.png",
)
