# -*- coding: utf-8 -*-
"""
中性策略框架 | 邢不行 | 2024分享会
author: 邢不行
微信: xbx6660
"""


def signal(*args):
    df = args[0]
    n = args[1]
    factor_name = args[2]

    df[factor_name] = df['quote_volume'].rolling(n, min_periods=1).mean()

    return df


    df['pos_return'] = df['close'].pct_change().apply(lambda x: x if x > 0 else 0)
    df['pos_quote_volume'] = np.where(df['pos_return'] > 0, df['quote_volume'], 0)
    df['最短路径_标准化_pos'] = np.where(df['pos_return'] > 0, df['盘中最短路径'] / df['open'], 0)
    df['流动溢价'] = np.where(df['pos_return'] > 0, df['pos_quote_volume'] / df['最短路径_标准化_pos'], 0)

