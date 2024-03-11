import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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

    df['ma1'] = df['close'].rolling(n, min_periods=1).mean()
    df['ma2'] = df['close'].rolling(2 * n, min_periods=1).mean()
    df['ma3'] = df['close'].rolling(4 * n, min_periods=1).mean()
    df['ma4'] = df['close'].rolling(8 * n, min_periods=1).mean()
    df['bbi'] = (df['ma1'] + df['ma2'] + df['ma3'] + df['ma4']) / 4

    df['bbi_bias'] = df['close'] / df['bbi']
    # df['bbi_bias'] = df['bbi_bias'].apply(lambda x: sigmoid(x))

    df[factor_name] = df['流动溢价'].rolling(n, min_periods=2).std() * df['bbi_bias']

    del df['route_1'], df['route_2'], df['盘中最短路径'], df['最短路径_标准化'], df['流动溢价'], df['ma1'], df['ma2'], df['ma3'], df['ma4'], df['bbi'], df['bbi_bias']

    return df
