#!/usr/bin/python3
# -*- coding: utf-8 -*-


def signal(*args):
    df = args[0]
    n = args[1]  # 观察期天数
    factor_name = args[2]

    # 计算未来N日的收益率和波动率
    returns = df['close'].pct_change()
    df['future_returns'] = returns.shift(-n).rolling(window=n).mean()
    df['future_volatility'] = returns.rolling(window=n).std().shift(-n)

    # 计算未来N日的夏普率
    df[factor_name] = df['future_returns'] / df['future_volatility']

    return df


def get_parameter():
    param_list = []
    n_list = [1, 2, 3, 6, 12, 24]
    for n in n_list:
        param_list.append(n)

    return param_list
