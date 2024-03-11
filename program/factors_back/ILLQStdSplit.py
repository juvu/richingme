# -*- coding: utf-8 -*-
"""
保温杯中性策略3期 | 邢不行 | 2023分享会
author: 邢不行
微信: xbx6660
"""
import numpy as np
import pandas as pd


def signal(*args):
    df = args[0]
    n = args[1]
    factor_name = args[2]

    df['route_1'] = 2 * (df['high'] - df['low']) + (df['open'] - df['close'])
    df['route_2'] = 2 * (df['high'] - df['low']) + (df['close'] - df['open'])
    df.loc[df['route_1'] > df['route_2'], '盘中最短路径'] = df['route_2']
    df.loc[df['route_1'] <= df['route_2'], '盘中最短路径'] = df['route_1']
    df['最短路径_标准化'] = df['盘中最短路径'] / df['open']
    df['流动溢价'] = df['quote_volume'] / df['最短路径_标准化']

    df[factor_name] = df['流动溢价'].rolling(n, min_periods=2).std()

    # 判断处于上涨通道还是下跌通道
    df['trend'] = np.where(df['close'].diff() > 0, 'up', 'down')

    # 分别计算上涨和下跌通道下的流动性
    df_up = df[df['trend'] == 'up'].copy()
    df_down = df[df['trend'] == 'down'].copy()

    df_up[factor_name] = 10000 / df_up['流动溢价'].rolling(n, min_periods=2).std()  # 上涨通道下的流动性, 流动性小的时候, 流动性因子大
    df_down[factor_name] = -df_down['流动溢价'].rolling(n, min_periods=2).std()  # 下跌通道下的流动性, 流动性大的时候, 流动性因子大

    df = pd.concat([df_up, df_down], axis=0).sort_index()

    return df
