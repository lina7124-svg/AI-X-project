# 뇌전도 기록을 통한 시각 자극 판별
> Option A

## Members
- 박예영, 의예과 1학년
- 변서현, 의예과 1학년, sunnybyeon@hanyang.ac.kr

## I. Proposal
### Motivation

 뇌파를 분석하여 생각을 추론할 수 있는 기술은 범죄 수사, 마케팅, 의료 등 다양한 분야에서 유용하게 사용될 수 있다. 뇌파를 분석할 때 측정할 수 있는 뇌의 부위와 모델에는 여러가지가 있다. 그중 어떤 부위를 측정했을 때, 그리고 어떤 모델을 적용했을 때 뇌파를 통한 시각 자극 예측을 더 정확하게 수행할 수 있는지 판단한다면, 앞으로 뇌파 연구를 할 때 더 효과적인 방법을 사용할 수 있을 것이라는 생각이 들었다. 
 0에서 9까지 숫자에 따른 FP1, AF3, AF4 각 부위의 뇌파 데이터 셋을 활용하여, 역으로 어떤 숫자를 본 것인지 추론하고 각각의 예측 정확도를 비교할 것이다. 또, 각 데이터 셋에 대해 CNN, LSTM, CNN-LSTM 모델을 적용해보고, 어떤 모델을 사용했을 때 가장 정확한 추론이 가능한지 분석해볼 것이다. 이를 통해, 뇌파를 통한 시각 자극 예측을 가장 정확하게 수행할 수 있는 방식을 알아보고자 하였다.

### **What do you want to see at the end?**

뇌의 각 위치의 시각 자극(숫자) 판별 성능(Accuracy, Loss)을 비교하고, 이를 통해 시각 자극 판별의 key가 되는 부위가 어디인지 알아낼 것이다. 또, 각 부위에 대해 CNN, LSTM, CNN-LSTM 모델을 적용하고 예측 정확도를 비교 분석하여, 어떤 모델이 뇌파를 통한 시각 자극 추론에 가장 효과적인지 알아볼 것이다.

## Ⅱ. Datasets

[MindBigData: The "MNIST" of Brain Digits on Kaggle](https://www.kaggle.com/datasets/vijayveersingh/1-2m-brain-signal-data)

위의 링크에서 확인할 수 있는 MindBigData를 사용하였다. 이 데이터셋은 한 사람을 대상으로 0에서 9까지의 숫자를 보여줬을 때 2초간의 뇌전도(EEG) 데이터를 담고 있다. Emotive EPOC (EP), Emotiv Insight (IN), Interaxon Muse (MU), NeuroSky MindWave (MW)의 4개의 기기를 이용해 측정한 데이터이며, 각 기기별로 제공된 데이터의 수는 다음과 같다. 여기서 데이터 수는 하나의 숫자를 보여주는 사건인 이벤트(event)의 수가 아니라, 각 이벤트의 채널(channel, 뇌전도 측정 위치)별 데이터를 모두 개별 데이터로 간주한 숫자이다.

| 기기 | 데이터 수 |
| ---- | --------- |
| EP   | 910,476   |
| IN   | 65,250    |
| MU   | 163,932   |
| MW   | 67,635    |

각 기기별로 다른 조합의 채널에서 데이터가 측정되었으며, 각 기기의 데이터 예시를 아래 사진에 나타내었다. 코드(code)는 대상자에게 보여준 숫자를 의미한다.

![EP](./figures/eeg-EP.png)
![IN](./figures/eeg-IN.png)
![MU](./figures/eeg-MU.png)
![MW](./figures/eeg-MW.png)

데이터 중 숫자가 아닌 자극을 의미하는 `-1`의 코드를 가진 데이터는 제외하였다. 또, 한 개의 기기에서만 측정한 채널은 제외하여 AF3, AF4, FP1의 3개 채널 데이터만 활용하였다. 이들 데이터의 코드별 분포는 다음과 같다.

![class-counts](./figures/class-count.png)

각 데이터는 채널별로 분류하여 다음의 전처리 과정을 거쳤다. 뇌전도 데이터가 같은 길이(시간 간격)로 제공되지 않아, 모델에 따라 필요 시 데이터의 마지막에 0을 추가해 길이를 통일하는 과정이 있었다.

```python
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
```

데이터 처리 및 시각화와 관련된 코드는 [dataset.py](./dataset.py)에서 확인할 수 있다.
