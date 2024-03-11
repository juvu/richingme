# -*- coding: utf-8 -*-
"""
轮动策略框架 | 邢不行 | 2024分享会
author: 邢不行
微信: xbx6660
"""


def signal(*args):
    df = args[0]
    n = args[1]
    factor_name = args[2]

    df['ma'] = df['close'].rolling(n, min_periods=1).mean()
    df['biasMean'] = df['close'] / df['ma']-1
    df[factor_name] = df['biasMean'].rolling(n, min_periods=1).mean()

    del df['ma'], df['biasMean']

    return df
