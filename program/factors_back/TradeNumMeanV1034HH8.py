# -*- coding: utf-8 -*-
"""
中性策略框架 | 邢不行 | 2024分享会
author: 邢不行
微信: xbx6660
"""
import pandas as pd

def signal(*args):
    df = args[0]
    n = args[1]
    factor_name = args[2]

    df['route_1'] = 2 * (df['high'] - df['low']) + (df['open'] - df['close'])
    df['route_2'] = 2 * (df['high'] - df['low']) + (df['close'] - df['open'])
    df.loc[df['route_1'] > df['route_2'], '盘中最短路径'] = df['route_2']
    df.loc[df['route_1'] <= df['route_2'], '盘中最短路径'] = df['route_1']
    df['ILLQ'] = (df['open'] / df['盘中最短路径']) * df['quote_volume']

    ma1 = df['close'].rolling(n, min_periods=1).mean()
    ma2 = df['close'].rolling(2 * n, min_periods=1).mean()
    ma3 = df['close'].rolling(4 * n, min_periods=1).mean()
    ma4 = df['close'].rolling(8 * n, min_periods=1).mean()
    bbi = (ma1 + ma2 + ma3 + ma4) / 4
    df['BbiBias'] = df['close'] / (bbi)

    # 使用滚动窗口计算因子的标准差，作为最终的因子值
    df['tmp'] = df['ILLQ'].rolling(n, min_periods=1).std() * df['BbiBias']

    original_index = df.index  # 保存原始索引
    df['datetime'] = pd.to_datetime(df['candle_begin_time'])
    # 将'candle_begin_time'设置为索引
    df.set_index('datetime', inplace=True)
    # 使用resample按4小时分组，并使用transform填充每个分组内的所有值为该组第一个值
    df[factor_name] = df['tmp'].resample('8H').transform('first')
    # 操作完成后，恢复原始索引
    df.reset_index(inplace=True)  # 仅在不需要'candle_begin_time'作为列时使用drop=True
    df.index = original_index  # 将保存的原始索引重新赋值给df

    # 清理中间生成的列，仅保留最终的因子值
    df.drop(['route_1', 'route_2', '盘中最短路径', 'BbiBias', 'tmp'], axis=1, inplace=True)

    return df

