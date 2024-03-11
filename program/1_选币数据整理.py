# -*- coding: utf-8 -*-
"""
中性策略框架 | 邢不行 | 2024分享会
author: 邢不行
微信: xbx6660
"""
import gc
import os.path
import time
from glob import glob
from Config import *
from Functions import *
import pandas as pd
import warnings

pd.set_option('display.max_rows', 1000)
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
warnings.filterwarnings("ignore")

# 动态读取config里面配置的strategy_name脚本
Strategy = __import__('strategy.%s' % strategy_name, fromlist=('',))


# =====处理每个币的数据
def calc_factors(file_path, symbol_type):
    print(file_path)
    # ===从文件路径中提取币种名
    symbol = os.path.basename(file_path).split('-USDT.csv')[0]

    # ===跳过稳定币、杠杆代币
    if symbol.endswith(('UP', 'DOWN', 'BEAR', 'BULL')) and symbol != 'JUP' or symbol in stable_symbol:
        print(symbol, '属于不参与交易的币种，直接跳过')
        return pd.DataFrame()

    # ===读取数据文件
    df = pd.read_csv(file_path, encoding='gbk', skiprows=1, parse_dates=['candle_begin_time'])
    df['开盘时间'] = df.iloc[0]['candle_begin_time']

    # ===与benchmark合并
    df = pd.merge(left=benchmark, right=df, on='candle_begin_time', how='left', sort=True, indicator=True)
    # 数据填充
    df['close'] = df['close'].fillna(method='ffill')
    df['open'] = df['open'].fillna(df['close'])
    df['high'] = df['high'].fillna(df['close'])
    df['low'] = df['low'].fillna(df['close'])
    df['开盘时间'] = df['开盘时间'].fillna(method='ffill')
    df['volume'] = df['volume'].fillna(0)
    df['quote_volume'] = df['quote_volume'].fillna(0)
    df['trade_num'] = df['trade_num'].fillna(0)
    df['taker_buy_base_asset_volume'] = df['taker_buy_base_asset_volume'].fillna(0)
    df['taker_buy_quote_asset_volume'] = df['taker_buy_quote_asset_volume'].fillna(0)
    df['symbol'] = df['symbol'].fillna(method='ffill')
    df['avg_price_1m'] = df['avg_price_1m'].fillna(df['open'])
    df['avg_price_5m'] = df['avg_price_5m'].fillna(df['open'])
    df['Spread'] = df['Spread'].fillna(method='ffill')
    # ===处理原始数据
    # 按照惯例排序并去重，防止原始数据有问题
    df.sort_values(by='candle_begin_time', inplace=True)
    df.drop_duplicates(subset=['candle_begin_time'], inplace=True, keep='last')
    # 计算需要的数据
    df['avg_price'] = df['avg_price_1m']  # 假设资金量很大，可以使用5m均价作为开仓均价
    df['avg_price'].fillna(value=df['open'], inplace=True)  # 若均价空缺，使用open填充
    df['下个周期_avg_price'] = df['avg_price'].shift(-1)  # 用于后面计算当周期涨跌幅
    df['是否交易'] = np.where(df['volume'] > 0, 1, 0)  # 存在成交量的数据，标识可交易状态。一个币种1h之内没有一笔交易，去除。但如果将来处理分钟数据，此处要改。
    df.reset_index(drop=True, inplace=True)

    # ===合并benchmark之后调用的方法：主要用于引入其他需要外部数据、设置外部数据在resample时需要进行的处理等。
    factor_dict, data_dict = {}, {}  # 创建空字典用于resample，factor_dict仅在转换日线的时候使用，data_dict是在转换到交易周期的时候使用
    df, factor_dict, data_dict = Strategy.after_merge_index(df, symbol, factor_dict, data_dict)
    # 合并其他数据之后，如果df为空，则处理下一个币种
    if df.empty:
        return pd.DataFrame()

    # ===计算选币因子
    # 此处会计算factors文件夹中所有py文件包含的因子。
    # 通过持仓周期来选择计算小时级别因子还是日线级别因子

    # =计算小时级别的选币因子
    if hold_period[-1] == 'H':  # 小时级别持仓，直接计算小时级别因子
        # df, factor_column_list = calc_factors_for_filename(df, factor_class_list, filename='factors', param_list=factor_param_list)
        # # 保存因子数据
        # save_factor(df, factor_class_list, root_path, symbol_type, symbol, factor_period='H')
        df, factor_column_list = calc_save_factors_for_filename(df, factor_class_list, root_path, symbol_type, symbol, filename='factors', param_list=factor_param_list, factor_period='H')
    # =计算日线级别的选币因子
    else:
        # 日线持仓，需要进行数据转换，并且计算日线的因子
        df_d = trans_period_for_day(df, factor_dict=factor_dict)
        # # 计算日线级别的选币因子
        # df_d, factor_column_list = calc_factors_for_filename(df_d, factor_class_list, filename='factors', param_list=factor_param_list)
        # # 保存因子数据
        # save_factor(df_d, factor_class_list, root_path, symbol_type, symbol, factor_period='D')
        df_d, factor_column_list = calc_save_factors_for_filename(df_d, factor_class_list, root_path, symbol_type, symbol, filename='factors', param_list=factor_param_list, factor_period='D')

        # # 取出计算得到的因子列
        # df_d = df_d[['candle_begin_time'] + factor_column_list]
        # # 将日线因子合并到小时级别数据中去
        # df = pd.merge(left=df, right=df_d, on='candle_begin_time', how='left')
        # 清理数据
        del df_d

    # ===计算涨跌幅
    df['开盘买入涨跌幅'] = df['close'] / df['avg_price'] - 1
    df['开盘卖出涨跌幅'] = df['下个周期_avg_price'] / df['close'].shift(1) - 1  # 必须用前收盘，否则k线不连续(当前一根k线的开盘价与上一根k线的收盘价差额较大)，就会出现计算误差
    df['涨跌幅'] = df['close'].pct_change(1)
    # ===删除数据中最后面的空数据
    df['avg_price'].fillna(method='bfill', inplace=True)  # 数据往前填充，主要是用于保留benchmark合并的数据，防止周期转换不一致
    df.dropna(subset=['avg_price'], inplace=True)  # 删除掉k线数据中最后面的空数据。主要原因是如果回测时间填写错误导致最后一个周期全是空数据，后面制作小时级别资金曲线会出问题

    # ===进行周期转换
    agg_dict = {'是否交易': 'first', 'close': 'last', 'avg_price': 'first', '下个周期_avg_price': 'last', '开盘买入涨跌幅': 'first',
                '开盘卖出涨跌幅': 'last', '开盘时间': 'last'}
    # 对每个因子设置转换规则
    # for f in list(set(factor_column_list)):
    #     agg_dict[f] = 'first'
    data_dict = dict(agg_dict, **data_dict)

    # ===在合约上架时间范围内，在现货数据中标记tag=HasSwap
    df, data_dict = get_swap_tag(df, symbol, symbol_type, swap_path, min_kline_num, data_dict, special_symbol_dict)

    # ===根据持仓周期进行转换
    df = trans_period_for_period(df, hold_period, data_dict=data_dict)

    # ===过滤掉开盘时间较少的币种
    df = df[df['candle_begin_time'] > df['开盘时间'] + pd.to_timedelta('1h') * min_kline_num]

    # ===周期转换之后调用的方法
    df = Strategy.after_resample(df, symbol)
    df.reset_index(inplace=True, drop=True)

    # ===保存数据
    header_cols = ['candle_begin_time', 'symbol']  # 头部字段
    save_cols = list(set(df.columns) - set(factor_column_list) - set(header_cols))
    save_cols.sort()  # 排序
    save_cols = header_cols + save_cols  # 将candle_begin_time，symbol字段放在最前面
    df = df[save_cols]
    # 构建存储周期数据目录路径
    save_file_path = os.path.join(root_path, 'data', '数据整理', symbol_type, symbol, 'periods')
    # 目录不存在，则创建目录
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)  # 创建目录

    for offset in list(range(0, int(hold_period[:-1]))):
        # 构建存储周期文件路径
        _file_path = os.path.join(save_file_path, f'{symbol}_{hold_period}_{offset}.pkl')
        # 文件不存在，则保存。避免重复计算，加快速度
        # if not os.path.exists(save_file_path):
        _df = df[df['offset'] == offset]
        _df.reset_index(inplace=True, drop=True)
        _df.to_feather(_file_path)

    # ===清理内存数据
    del df, _df
    gc.collect()


if __name__ == '__main__':
    """
    文件改动较小，但是拆分比较琐碎。
    对于同一个周期的数据和因子只会计算一次
    如果修改了回测结束时间，请重新计算
    """
    s_time = time.time()
    # =====准备工作
    # 获取Strategy的hold_period，后续使用
    hold_period = Strategy.hold_period
    # 获取Strategy的if_use_spot
    if_use_spot = Strategy.if_use_spot

    # =====获取所有文件路径
    # 获取所有合约文件的路径
    swap_symbol_path = glob(swap_path + '*USDT.csv')  # 获取kline_path路径下，所有以usdt.csv结尾的文件路径
    swap_file_path = []
    for _symbol_path in swap_symbol_path:
        swap_file_path.append([_symbol_path, 'swap'])
    # 获取所有现货文件的路径
    spot_file_path = []
    if if_use_spot:
        spot_symbol_path = glob(spot_path + '*USDT.csv')  # 获取kline_path路径下，所有以usdt.csv结尾的文件路径
        for _symbol_path in spot_symbol_path:
            spot_file_path.append([_symbol_path, 'spot'])
    # 合并合约和现货文件路径信息
    symbol_file_path = swap_file_path + spot_file_path

    # =====并行或串行，依次读取每个币种的数据，进行处理，并最终合并成一张大表输出
    multiply_process = True  # 是否并行。在测试的时候可以改成False，实际跑的时候改成True
    if multiply_process:
        Parallel(n_jobs=n_jobs)(
            delayed(calc_factors)(file_path, symbol_type)
            for file_path, symbol_type in symbol_file_path)
    else:
        for file_path, symbol_type in symbol_file_path:
            calc_factors(file_path, symbol_type)

    print('数据计算完成！', time.time() - s_time)
