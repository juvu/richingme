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
    df['bias'] = df['close'] / df['ma']
    df[factor_name] = df['bias'].pct_change(n)

    del df['ma'], df['bias']

    return df
