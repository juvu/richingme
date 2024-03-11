#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def signal(*args):
    df = args[0]
    n = args[1]
    factor_name = args[2]

    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['ma'] = df['tp'].rolling(window=n, min_periods=1).mean()
    df['md'] = abs(df['close'] - df['ma']).rolling(window=n, min_periods=1).mean()
    df['cci'] = (df['tp'] - df['ma']) / (df['md'] * 0.015)

    original_index = df.index  # 保存原始索引
    # 构造'datetime'索引
    df['datetime'] = pd.to_datetime(df['candle_begin_time'])
    df.set_index('datetime', inplace=True)
    # 使用resample按4小时分组，并使用transform填充每个分组内的所有值为该组第一个值
    df[factor_name] = df['cci'].resample('8H').transform('first')
    # 操作完成后，恢复原始索引
    df.reset_index(drop=True, inplace=True)  # 仅在不需要'candle_begin_time'作为列时使用drop=True
    df.index = original_index  # 将保存的原始索引重新赋值给df

    df.drop(['tp', 'ma', 'md', 'cci', ], axis=1, inplace=True)
    return df

