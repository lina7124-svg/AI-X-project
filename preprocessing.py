import numpy as np
import pandas as pd


###### 데이터셋 읽기 ######


ep_path = "./data/MindBigData-EP-v1.0/EP1.01.txt"


def load_mindbigdata_ep(path, max_lines=None):
    rows = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            parts = line.strip().split("\t")
            if len(parts) < 7:
                continue

            trial_id = parts[0]
            timestamp = parts[1]
            device = parts[2]
            channel = parts[3]
            code = int(parts[4])
            size = int(parts[5])
            data_str = parts[6]

            if code == -1:  # -1인 경우 제외
                continue

            rows.append(
                {
                    "trial_id": trial_id,
                    "timestamp": timestamp,
                    "device": device,
                    "channel": channel,
                    "code": code,
                    "size": size,
                    "data_str": data_str,
                }
            )
    return pd.DataFrame(rows)


df_ep = load_mindbigdata_ep(ep_path)


def to_array(s):
    return np.array(s.split(","), dtype=float)


df_ep["data_arr_original"] = df_ep["data_str"].apply(to_array)
# data_str은 더이상 필요하지 않으니 drop
df_ep.drop("data_str", inplace=True)


# timestamp 기준으로 정렬
df_ep = df_ep.sort_values("timestamp").reset_index(drop=True)


# 데이터셋 확인
print(f"전체 데이터 형태: {df_ep.shape}")
print("----- 전체 데이터 일부 예시 -----")
print(df_ep.head())


# 채널 목록 확인
channels = sorted(df_ep["channel"].unique())
print("전체 채널:", channels)


###### 전체 전처리 ######


# EEG 데이터 길이 통일
def resize_signal(sig, target_len):
    cur = len(sig)
    x_old = np.linspace(0, 1, cur)
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, sig)


max_length = df_ep["size"].max()
df_ep["data_arr"] = df_ep["data_arr_original"].apply(
    lambda data: resize_signal(data, max_length)
)

print(f"전체 채널 통일 길이: {max_length}")


# EEG 데이터 Z-score 정규화
def normalize_signal(sig, eps=1e-8):
    sig = sig.astype(float)
    mean = sig.mean()
    std = sig.std()
    if std < eps:
        return sig - mean
    return (sig - mean) / std


df_ep["data_arr"] = df_ep["data_arr"].apply(normalize_signal)


###### 데이터셋 train과 validation으로 나누기 ######


# shuffle 후 48:1로 train과 val 데이터셋 나누기 (ERP용)
timestamps = df_ep["timestamp"].unique()
np.random.shuffle(timestamps)

split_index_erp = int(np.floor(len(timestamps) * (48 / 49)))
train_timestamps_erp = timestamps[:split_index_erp]
val_timestamps_erp = timestamps[split_index_erp:]

df_train_erp = df_ep[df_ep["timestamp"].isin(train_timestamps_erp)].copy()
df_val_erp = df_ep[df_ep["timestamp"].isin(val_timestamps_erp)].copy()

# print(f"Train for (ERP): {len(df_train_erp)}")
# print(f"Validation for (ERP): {len(df_val_erp)}")


# shuffle 후 8:2로 train과 val 데이터셋 나누기 (EEG, CNN용)
split_index = int(np.floor(len(timestamps) * 0.8))
train_timestamps = timestamps[:split_index]
val_timestamps = timestamps[split_index:]

df_train = df_ep[df_ep["timestamp"].isin(train_timestamps)]
df_val = df_ep[df_ep["timestamp"].isin(val_timestamps)]

# print(f"Train for (EEG, CNN): {len(df_train)}")
# print(f"Validation for (EEG, CNN): {len(df_val)}")


###### ERP 전처리 ######


def extract_erp(df, n_trials_per_erp=12, min_trials=8):
    erp_list = []

    for code in sorted(df["code"].unique()):
        subset = df[df["code"] == code].copy()

        signals = list(subset["data_arr"])
        idx = 0

        while idx < len(signals):
            chunk = signals[idx : idx + n_trials_per_erp]
            if len(chunk) < min_trials:
                break

            erp_signal = np.array(chunk).mean(axis=0)

            erp_list.append(
                {"code": int(code), "signal": erp_signal, "length": len(erp_signal)}
            )

            idx += n_trials_per_erp

    return pd.DataFrame(erp_list)


erp_dict = {}

for ch in channels:
    df_ch = df_train_erp[df_train_erp["channel"] == ch].copy()

    erp_dict[ch] = extract_erp(df_ch)
    # print(f"{ch} ERP: {erp_dict[ch].shape}")


def build_multi_channel_dataset(erp_dict, channels):
    X_list = []
    Y_list = []

    for code in range(10):
        per_ch = []
        for ch in channels:
            df_ch = erp_dict[ch]
            tmp = df_ch[df_ch["code"] == code].reset_index(drop=True)
            per_ch.append(tmp)

        n = min([len(df) for df in per_ch])

        for i in range(n):
            multi_ch = []
            for k, ch in enumerate(channels):
                sig = per_ch[k].loc[i, "signal"]
                multi_ch.append(sig)

            multi_ch = np.stack(multi_ch, axis=-1)  # (time, channels)
            X_list.append(multi_ch)
            Y_list.append(code)

    X = np.array(X_list)
    Y = np.array(Y_list)
    return X, Y


X_ERP_train, Y_ERP_train = build_multi_channel_dataset(erp_dict, channels)

# print("X_ERP:", X_ERP_train.shape)
# print("Y_ERP:", Y_ERP_train.shape)


X_ERP_val = np.array(
    df_val_erp.groupby("timestamp")["data_arr"]
    .apply(lambda x: np.stack(x, axis=1))
    .to_list()
)
Y_ERP_val = np.array(df_val_erp.groupby("timestamp")["code"].first().to_list())


###### Recurrence Plot 전처리 ######


# data = (channels, data_length)
# assumes data_length is same for all channels
def to_recurrence_plot(data):
    plots = []
    for channel_data in data:
        data_length = len(channel_data)

        plot = np.zeros((data_length, data_length))
        for i in range(data_length):
            for j in range(data_length):
                plot[i, j] = np.abs(channel_data[i] - channel_data[j])

        plots.append(plot)

    result = np.sum(plots, axis=0)

    # normalizing the plot
    result = (result - result.min()) / (result.max() - result.min())

    return result
