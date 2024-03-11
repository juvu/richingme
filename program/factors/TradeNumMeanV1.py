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

    df['振幅'] = (df['high'] - df['low']) / df['open']
    df['factor'] = df['振幅'] / df['trade_num']

    df[factor_name] = df['factor'].rolling(n, min_periods=1).mean()

    return df
