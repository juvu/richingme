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

    df[factor_name] = df['close'].pct_change(n)

    return df


def get_parameter():
    param_list = []
    n_list = [7]
    for n in n_list:
        param_list.append(n)

    return param_list
