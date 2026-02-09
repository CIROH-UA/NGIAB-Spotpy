from typing import DefaultDict

import pandas as pd
from scipy.stats import circmean


def weighted_mode(df: pd.DataFrame, weights: dict[str, float]):
    mode = DefaultDict(float, default=0)
    for id in df.index:
        value = df.loc[id]
        weight = weights.get(id, 0)
        mode[value] += weight
    return max(mode.items(), key=lambda x: x[1])[0]


def weighted_circ_mean(df: pd.DataFrame, weights: dict[str, float]):
    # this is probably very slow for large datasets
    # optimize if needed
    # round everything to 2 decimal places then multiply by 100 so that the result is an integer
    # catid areasqkm aspect
    # 1     2    100
    # 2     1    50
    # 3     5    10
    # [100, 100, 50, 10, 10, 10, 10, 10]
    # 5.12342 sqkm becomes 510 weight
    # 5.12 sqkm
    # 512

    adjusted_weights = {}
    for key, value in weights.items():
        adjusted_weights[key] = round(value, 2) * 100
    samples = []
    for id in df.index:
        value = df.loc[id]
        weight = adjusted_weights.get(id, 0)
        for i in range(int(weight)):
            samples.append(value)
    return circmean(samples, high=360)
