# -*- coding: utf-8 -*-
"""
中性策略框架 | 邢不行 | 2024分享会
author: 邢不行
微信: xbx6660
"""
import numpy as np
import pandas as pd
import os
from glob import glob
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 1000)
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行


def get_file_in_folder(path, file_type, contains=None, filters=[], drop_type=False):
    """
    获取指定文件夹下的文件
    :param path: 文件夹路径
    :param file_type: 文件类型
    :param contains: 需要包含的字符串，默认不含
    :param filters: 字符串中需要过滤掉的内容
    :param drop_type: 是否要保存文件类型
    :return:
    """
    file_list = os.listdir(path)
    file_list = [file for file in file_list if file_type in file]
    if contains:
        file_list = [file for file in file_list if contains in file]
    for con in filters:
        file_list = [file for file in file_list if con not in file]
    if drop_type:
        file_list = [file[:file.rfind('.')] for file in file_list]

    return file_list


def calc_factors_for_filename(df, factor_list, filename='', param_list=[]):
    """
    使用文件夹下的因子脚本进行计算因子
    :param df: 原始k线数据
    :param factor_list: 需要计算的因子列表
    :param filename: 指定因子文件夹名称
    :param param_list: 因子参数
    :return:
    """
    column_list = []
    # 根据config中设置的因子列表，逐个计算每个因子的数据
    for factor in factor_list:
        factor = factor.split('.')[0]
        _cls = __import__('%s.%s' % (filename, factor), fromlist=('',))
        # 兼容get_parameter
        if 'get_parameter' in _cls.__dict__:  # 如果存在get_parameter，直接覆盖config中配置的factor_param_list
            _param_list = getattr(_cls, 'get_parameter')()
        else:  # 如果不存在get_parameter，直接用config中配置的factor_param_list
            _param_list = param_list.copy()
        # 遍历参数，计算每个参数对应的因子值
        for n in _param_list:
            # 构建因子列名
            factor_name = f'{factor}_{str(n)}'
            # 因子已经计算过，则跳过
            if factor_name in df.columns:
                continue
            # 计算因子
            df = getattr(_cls, 'signal')(df, n, factor_name)
            # 为了跟实盘保持一致，所有因子信息在下个周期生效
            df[factor_name] = df[factor_name].shift(1)
            # 保存因子列名
            column_list.append(factor_name)

    return df, column_list


# 把以下函数添加到现有保温杯function.py中即可
def calc_save_factors_for_filename(df, factor_list,root_path,symbol_type,symbol, filename='', param_list=[],factor_period='H'):
    """
    使用文件夹下的因子脚本进行计算因子，并且保存因子文件
    :param df: 原始k线数据
    :param factor_list: 需要计算的因子列表
    :param root_path: 导出整理数据的根目录位置
    :param symbol_type: 币种类型，spot/swap
    :param symbol: 币种名称
    :param filename: 指定因子文件夹名称
    :param param_list: 因子参数
    :param factor_period: 因子周期
    :return:
    """
    column_list = []
    # 根据config中设置的因子列表，逐个计算每个因子的数据
    for factor in factor_list:
        cfactor_column_list = []
        factor = factor.split('.')[0]
        _cls = __import__('%s.%s' % (filename, factor), fromlist=('',))
        # 兼容get_parameter
        if 'get_parameter' in _cls.__dict__:  # 如果存在get_parameter，直接覆盖config中配置的factor_param_list
            _param_list = getattr(_cls, 'get_parameter')()
        else:  # 如果不存在get_parameter，直接用config中配置的factor_param_list
            _param_list = param_list.copy()
        # 遍历参数，计算每个参数对应的因子值
        for n in _param_list:
            # print(f"计算因子:{factor}_{str(n)}")
            # 构建因子列名
            factor_name = f'{factor}_{str(n)}'
            # 因子已经计算过，则跳过
            if factor_name in df.columns:
                continue
            # 计算因子
            df = getattr(_cls, 'signal')(df, n, factor_name)
            # 为了跟实盘保持一致，所有因子信息在下个周期生效
            df[factor_name] = df[factor_name].shift(1)
            # 保存因子列名
            column_list.append(factor_name)
            cfactor_column_list.append(factor_name)

        # print(f"保存因子")
        # 构建存储目录路径
        save_file_path = os.path.join(root_path, 'data', '数据整理', symbol_type, symbol, 'factors')
        # 目录不存，就创建目录
        if not os.path.exists(save_file_path):
            os.makedirs(save_file_path)  # 创建目录
        # 构建存储文件目录
        save_file_path = os.path.join(save_file_path, f'{symbol}_{factor}_{factor_period}.pkl')
        # 文件不存在，则保存。避免重复计算
        # if not os.path.exists(save_file_path):
        df_f = df.dropna(subset=cfactor_column_list, how='all')
        df_f.reset_index(inplace=True)
        df_f[['candle_begin_time'] + cfactor_column_list].to_feather(save_file_path)

        df = df.drop(cfactor_column_list,axis=1)

    return df, column_list


def trans_period_for_period(df, period, data_dict=None):
    """
    周期转换函数
    :param df: K线数据
    :param period: 数据转换周期
    :param data_dict: 转换规则
    """
    # ===数据整理
    df.set_index('candle_begin_time', inplace=True)  # 重置index
    df['fundingRate'].fillna(value=0.0, inplace=True)  # 资金费数据用0填充

    # ===转换必备字段
    agg_dict = {
        'symbol': 'first',
        '_merge': 'first'
    }
    if data_dict:
        agg_dict = dict(agg_dict, **data_dict)

    if period == '1H':
        period_df = df[agg_dict.keys()]
        period_df['每小时涨跌幅'] = df['涨跌幅'].transform(lambda x: [x])
        period_df['fundingRate'] = df['fundingRate'].transform(lambda x: [x])
        period_df['offset'] = 0
        period_df.reset_index(inplace=True)

        # 删除nan
        period_df.dropna(subset=['symbol'], inplace=True)
        period_df = period_df[period_df['_merge'] == 'both']  # 删除掉周期不全数据，有些币重新上线之后前面的数据周期不全，直接删除
        del period_df['_merge']
    else:
        # ===数据转换
        period_df_list = []
        # 通过持仓周期来计算需要多少个offset，遍历转换每一个offset数据
        for offset in range(int(period[:-1])):
            period_df = df.resample(period, base=offset).agg(agg_dict)
            period_df['每小时涨跌幅'] = df['涨跌幅'].resample(period, base=offset).apply(lambda x: list(x))
            period_df['fundingRate'] = df['fundingRate'].resample(period, base=offset).apply(lambda x: list(x))
            period_df['offset'] = offset
            period_df.reset_index(inplace=True)

            # 删除nan
            period_df.dropna(subset=['symbol'], inplace=True)
            period_df = period_df[period_df['_merge'] == 'both']  # 删除掉周期不全数据，有些币重新上线之后前面的数据周期不全，直接删除
            del period_df['_merge']

            # 数据存放到list中
            period_df_list.append(period_df)

        # ===数据合并
        # 将不同offset的数据，合并到一张表
        period_df = pd.concat(period_df_list, ignore_index=True)

    period_df.sort_values(by='candle_begin_time', inplace=True)
    period_df.reset_index(drop=True, inplace=True)

    return period_df


def get_swap_tag(df, symbol, symbol_type, swap_path, min_kline_num, exg_dict, special_symbol_dict):
    """
    在数据中加入tag列，表明现货是否拥有对应的合约
    :param df:
    :param symbol: 币种名
    :param symbol_type: K线类型，是spot还是swap
    :param swap_path: 合约数据路径
    :param min_kline_num: 最少上市多久
    :param exg_dict: resample时指定的字典
    :param special_symbol_dict: 特殊现货对应合约关系
    :return:
    """
    # 新增symbol_type列：数据类型(spot、swap)
    df['symbol_type'] = symbol_type
    # 新增tag列：是否拥有合约，没有为None，有为HasSwap
    df['tag'] = None

    # 判断数据类型是否为spot，只有spot才需要标记HasSwap
    if symbol_type == 'spot':
        # 现货的资金费率为0
        df['fundingRate'] = 0
        # 拼接出该币种合约数据的路径
        _symbol = special_symbol_dict.get(symbol, symbol)  # 兼容币种别名的
        _swap_file = os.path.join(swap_path, _symbol + '-USDT.csv')
        _swap_file2 = os.path.join(swap_path, '1000' + _symbol +'-USDT.csv')
        # 如果存在合约数据的，在合约的上时间范围内标记HasSwap
        if os.path.exists(_swap_file) or os.path.exists(_swap_file2):
            try:
                _swap_df = pd.read_csv(_swap_file, encoding='gbk', skiprows=1, parse_dates=['candle_begin_time'])
            except:
                _swap_df = pd.read_csv(_swap_file2, encoding='gbk', skiprows=1, parse_dates=['candle_begin_time'])
            cond1 = df['candle_begin_time'] > _swap_df.iloc[0]['candle_begin_time'] + pd.to_timedelta('1H') * (min_kline_num + 24)
            cond2 = df['candle_begin_time'] <= _swap_df.iloc[-1]['candle_begin_time']
            df.loc[cond1 & cond2, 'tag'] = 'HasSwap'

        # 合约有改名的情况 ========================================
        if _symbol != symbol:  # 兼容币种改名。LUNA -> LUNA2。2022-5-12之前还是LUNA，2022-9-10改名为LUNA2
            _swap_file = os.path.join(swap_path, symbol + '-USDT.csv')
            _swap_file2 = os.path.join(swap_path, '1000' + symbol +'-USDT.csv')
            # 如果存在合约数据的，在合约的上时间范围内标记HasSwap
            if os.path.exists(_swap_file) or os.path.exists(_swap_file2):
                try:
                    _swap_df = pd.read_csv(_swap_file, encoding='gbk', skiprows=1, parse_dates=['candle_begin_time'])
                except:
                    _swap_df = pd.read_csv(_swap_file2, encoding='gbk', skiprows=1, parse_dates=['candle_begin_time'])
                cond1 = df['candle_begin_time'] > _swap_df.iloc[0]['candle_begin_time'] + pd.to_timedelta('1H') * (min_kline_num + 24)
                cond2 = df['candle_begin_time'] <= _swap_df.iloc[-1]['candle_begin_time']
                df.loc[cond1 & cond2, 'tag'] = 'HasSwap'
        # 合约有改名的情况 ========================================

    # 指定symbol_type列、tag列 转换周期的参数
    exg_dict['symbol_type'] = 'last'
    exg_dict['tag'] = 'first'

    return df, exg_dict


def trans_period_for_day(df, factor_dict=None):
    """
    将数据周期转换为指定的1D周期，不计算
    :param df: K线数据
    :param factor_dict: 转换规则
    :return:
    """
    # 拷贝一份数据，转换成日线，计算日线的因子
    df_d = df.copy()
    df_d.set_index('candle_begin_time', inplace=True)
    # 必备字段
    agg_dict = {
        'symbol': 'first',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'quote_volume': 'sum',
        'volume': 'sum',
        '_merge': 'first',
    }
    if factor_dict:
        agg_dict = dict(agg_dict, **factor_dict)
    df_d = df_d.resample('1D').agg(agg_dict)
    df_d.sort_values(by='candle_begin_time', inplace=True)
    df_d.reset_index(inplace=True)

    return df_d


def create_empty_data(index_data, period, offset=0):
    """
    创建空的周期，用于填充未开仓的数据
    :param index_data:  基准数据
    :param period:      持仓周期
    :param offset:      offset偏移周期
    :return:
    """
    # 根据index的时间数据构建df
    empty_df = index_data[['candle_begin_time']].copy()
    empty_df['涨跌幅'] = 0.0
    empty_df['多头涨跌幅'] = 0.0
    empty_df['空头涨跌幅'] = 0.0

    empty_df['涨跌幅_byclose'] = 0.0
    empty_df['多头涨跌幅_byclose'] = 0.0
    empty_df['空头涨跌幅_byclose'] = 0.0

    empty_df.set_index('candle_begin_time', inplace=True)

    # 构建空的周期数据
    empty_period_df = pd.DataFrame()
    empty_period_df['每小时涨跌幅'] = empty_df['涨跌幅'].resample(period, base=offset).apply(lambda x: list(x))
    empty_period_df['多头每小时涨跌幅'] = empty_df['多头涨跌幅'].resample(period, base=offset).apply(lambda x: list(x))
    empty_period_df['空头每小时涨跌幅'] = empty_df['空头涨跌幅'].resample(period, base=offset).apply(lambda x: list(x))
    # 填充其他列
    empty_period_df['offset'] = np.nan
    empty_period_df['选币'] = np.nan
    empty_period_df['每小时资金曲线'] = empty_period_df['每小时涨跌幅'].transform(lambda x: np.ones(len(x)))
    empty_period_df['多头每小时资金曲线'] = empty_period_df['多头每小时涨跌幅'].transform(lambda x: np.ones(len(x)))
    empty_period_df['空头每小时资金曲线'] = empty_period_df['空头每小时涨跌幅'].transform(lambda x: np.ones(len(x)))
    empty_period_df['周期涨跌幅'] = 0.0
    empty_period_df['多头周期涨跌幅'] = 0.0
    empty_period_df['空头周期涨跌幅'] = 0.0
    empty_period_df['多空调仓比例'] = 0.0
    empty_period_df['多头调仓比例'] = 0.0
    empty_period_df['空头调仓比例'] = 0.0

    empty_period_df['每小时涨跌幅_byclose'] = empty_df['涨跌幅_byclose'].resample(period, base=offset).apply(lambda x: list(x))
    empty_period_df['每小时资金曲线_byclose'] = empty_period_df['每小时涨跌幅_byclose'].transform(lambda x: np.ones(len(x)))
    empty_period_df['周期涨跌幅_byclose'] = 0.0
    empty_period_df['多头每小时涨跌幅_byclose'] = empty_df['多头涨跌幅_byclose'].resample(period, base=offset).apply(lambda x: list(x))
    empty_period_df['多头每小时资金曲线_byclose'] = empty_period_df['多头每小时涨跌幅_byclose'].transform(lambda x: np.ones(len(x)))
    empty_period_df['多头周期涨跌幅_byclose'] = 0.0
    empty_period_df['空头每小时涨跌幅_byclose'] = empty_df['空头涨跌幅_byclose'].resample(period, base=offset).apply(lambda x: list(x))
    empty_period_df['空头每小时资金曲线_byclose'] = empty_period_df['空头每小时涨跌幅_byclose'].transform(lambda x: np.ones(len(x)))
    empty_period_df['空头周期涨跌幅_byclose'] = 0.0

    # 筛选指定字段
    empty_period_df = empty_period_df[['offset', '选币', '每小时资金曲线', '周期涨跌幅', '每小时涨跌幅', '多空调仓比例', '多头调仓比例', '空头调仓比例',
                                       '多头每小时资金曲线', '多头周期涨跌幅', '多头每小时涨跌幅', '空头每小时资金曲线', '空头周期涨跌幅', '空头每小时涨跌幅',
                                       '每小时涨跌幅_byclose', '每小时资金曲线_byclose', '周期涨跌幅_byclose',
                                       '多头每小时涨跌幅_byclose', '多头每小时资金曲线_byclose', '多头周期涨跌幅_byclose',
                                       '空头每小时涨跌幅_byclose', '空头每小时资金曲线_byclose', '空头周期涨跌幅_byclose'
                                       ]]
    return empty_period_df


def transfer_swap(select_coin, df_swap, special_symbol_dict):
    """
    将含有合约的现货币种数据，替换成为合约数据，计算合约的涨跌幅
    :param select_coin:         选币数据
    :param df_swap:             合约数据
    :param special_symbol_dict: 现货与合约对应关系
    :return:
    """
    # 反转k,v的对应关系。现货对应合约，转换 ，合约对应现货
    special_symbol_dict = {v: k for k, v in special_symbol_dict.items()}
    # 整理币种名称，使合约和现货的币种名称相同
    df_swap['symbol_spot'] = df_swap['symbol'].apply(lambda x: special_symbol_dict.get(x.replace('-USDT', ''), (x.split('1000')[1] if '1000' in x else x).replace('-USDT', '')) + '-USDT')
    # 将最终选币的结果数据分为两部分：有合约数据和无合约数据（其中空头选币一定为HasSwap）
    not_swap_df = select_coin[select_coin['tag'] != 'HasSwap']
    has_swap_df = select_coin[select_coin['tag'] == 'HasSwap']

    # 如果存在需要替换合约涨跌幅的现货币种，则使用对应合约币种的啊行情数据将现货的行情数据替换掉
    if not has_swap_df.empty:
        # 选币之后的数据相较于合约数据多出来的columns
        other_cols = list(set(select_coin.columns) - set(df_swap.columns))

        # =将有合约的现货数据和其对应的合约数据合并，
        # 此步操作后has_swap_df的[每小时涨跌幅]字段会变成合约的涨跌幅，symbol_type会变成swap
        # has_swap_df = pd.merge(has_swap_df[['candle_begin_time', 'symbol'] + other_cols], df_swap, on=['candle_begin_time', 'symbol'], how='left')
        has_swap_df = pd.merge(left=has_swap_df[['candle_begin_time', 'symbol'] + other_cols], right=df_swap,
                               left_on=['candle_begin_time', 'symbol'], right_on=['candle_begin_time', 'symbol_spot'],
                               how='left', suffixes=('', '_swap'))
        has_swap_df['symbol'] = has_swap_df['symbol_swap']
        del has_swap_df['symbol_swap'], has_swap_df['symbol_spot']
        # 删除[每小时涨跌幅]为空的数据
        has_swap_df.dropna(subset=['每小时涨跌幅'], axis=0, inplace=True)

    # 将替换后的合约行情数据与不需要替换的数据合并、整理数据
    select_coin = pd.concat([not_swap_df, has_swap_df], axis=0)
    select_coin.sort_values(['candle_begin_time', 'symbol'], axis=0, inplace=True)

    return select_coin


def calc_rank(df, factor_column='因子', ascending=True):
    """
    计算因子排名
    :param df:              原数据
    :param factor_column:   需要计算排名的因子名称
    :param ascending:       计算排名顺序，True：从小到达排序；False：从大到小排序
    :return:
    """
    # 计算因子的分组排名
    df['rank'] = df.groupby('candle_begin_time')[factor_column].rank(method='min', ascending=ascending)
    df['rank_max'] = df.groupby('candle_begin_time')['rank'].transform('max')
    # 根据时间和因子排名
    df.sort_values(by=['candle_begin_time', 'rank'], inplace=True)
    # 重新计算一下总币数
    df['总币数'] = df.groupby('candle_begin_time')['symbol'].transform('size')

    return df


def select_long_and_short_coin(long_df, short_df, long_select_coin_num, short_select_coin_num, long_factor='因子', short_factor='因子'):
    """
    选币
    :param long_df:                 多头选币的df
    :param short_df:                空头选币的df
    :param long_select_coin_num:    多头选币数量
    :param short_select_coin_num:   空头选币数量
    :param long_factor:             做多因子名称
    :param short_factor:            做空因子名称
    :return:
    """
    # ===做多选币，因子值相同时全选
    long_df = calc_rank(long_df, factor_column=long_factor, ascending=True)
    # ？？？
    if int(long_select_coin_num) == 0:
        long_df = long_df[long_df['rank'] <= long_df['总币数'] * long_select_coin_num]
    else:
        long_df = long_df[long_df['rank'] <= long_select_coin_num]
    long_df['方向'] = 1

    # ===做空选币
    short_df = calc_rank(short_df, factor_column=short_factor, ascending=False)
    if short_select_coin_num == 'long_nums':  # 如果参数是long_nums，则空头与多头的选币数量保持一致
        # 获取到多头的选币数量并整理数据
        long_select_num = long_df.groupby('candle_begin_time')['symbol'].size().to_frame()
        long_select_num = long_select_num.rename(columns={'symbol': '多头数量'}).reset_index()
        # 将多头选币数量整理到short_df
        short_df = short_df.merge(long_select_num, on='candle_begin_time', how='left')
        # 使用多头数量对空头数据进行选币
        short_df = short_df[short_df['rank'] <= short_df['多头数量']]
    else:
        # 百分比选币
        if int(short_select_coin_num) == 0:
            short_df = short_df[short_df['rank'] <= short_df['总币数'] * short_select_coin_num]
        # 固定数量选币
        else:
            short_df = short_df[short_df['rank'] <= short_select_coin_num]
    short_df['方向'] = -1

    # ===整理数据
    df = pd.concat([long_df, short_df], ignore_index=True)   # 将做多和做空的币种数据合并
    df.sort_values(by=['candle_begin_time', '方向'], ascending=[True, False], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def cal_net_value(data, hold_period, trading_time_list):
    """
    根据换仓信息，精确计算手续费及资金费
    :param data: 输入的原始数据
    :param hold_period: 持仓周期
    :param trading_time_list: 交易时间
    :return:
    """
    if data.empty:
        return pd.DataFrame()
    # =====从输入的原始数据中导出当前持币信息，及上下周期的持币信息
    hold_df = pd.DataFrame(data.groupby(['candle_begin_time'])['symbol'].apply(lambda x: list(x))).reset_index()

    hold_df = pd.merge(left=trading_time_list, right=hold_df, on='candle_begin_time', how='left')
    hold_df['symbol'] = hold_df['symbol'].apply(lambda x: [] if not isinstance(x, list) and (np.isnan(x) or x is None) else x)

    # 获取上周期和下周期的持币信息
    hold_df['symbol_上周期'] = hold_df['symbol'].shift(1)
    hold_df['symbol_下周期'] = hold_df['symbol'].shift(-1)
    # 将空的数据用[]填充
    hold_df.at[0, 'symbol_上周期'] = []
    hold_df.at[hold_df.index.max(), 'symbol_下周期'] = []

    """
    估算小时频率的手续费
    为什么是估算？
    假设我们目前持仓只有AB两个币，下周期依旧持仓AB。
    开仓的时候AB的权重都是50%，下周期理论上AB的权重已经随着价格变化了，我们下周期要把AB的权重重新调整会50%
    但是我们忽略这个变化，所以是估算，即不考虑价格波动导致的权重变化以及由此引入的手续费。
    还有一个近似假设：多空的币数是一样的。

    下面来看三个例子
       当前持仓             下周期持仓                   卖出AB的比例            买入C比例
    A（1/2）B（1/2）      A（1/3）B（1/3）C（1/3）   （1/2-1/3）*2 = 1/3            1/3
       当前持仓                  下周期持仓                   卖出ABC的比例            买入DE比例
A（1/3）B（1/3）C（1/3）  A（1/4）B（1/4）D（1/4）E（1/4）   （1/3-1/4）*2+1/3= 1/2      1/4*2=1/2
       当前持仓                         下周期持仓              卖出ABDE的比例            买入ABC比例
A（1/4）B（1/4）D（1/4）E（1/4）   A（1/3）B（1/3）C（1/3）      0*2+1/4*2=1/2          （1/3-1/4）*2+1/3= 1/2 


    观察上面两个例子可以得到以下结论
    1、卖出比例 = 买入比例
    2、卖出比例 = 买入比例 = 1-（新旧持仓交集币数）/ max（旧持仓币数，新持仓币数）

    针对第一个例子
    比例 = 1 -（AB）/max(AB,ABC) = 1-2/3 = 1/3
    针对第二个例子
    比例 = 1 -（AB）/max(ABC,ABDE) = 1-2/4 = 1/2
    针对第三个例子
    比例 = 1 -（AB）/max(ABC,ABDE)  = 1-2/4 = 1/2

    特殊情况： 当新持仓为空时，算出1-inf，卖出强制设置为1，买入比例为0
    """

    # ===根据上面推导的公式计算买入卖出的比例
    def cal_change_pct(row, old, new):
        """
        计算调仓比例
        :param row: 每一行数据
        :param old: 上周期选币数据
        :param new: 新周期选币数据
        :return:
        """
        # 计算交集数量
        intersection = len(list(set(row[old]) & set(row[new])))
        # 计算 旧周期 & 新周期 的币数
        len_old = len(row[old])
        len_new = len(row[new])
        # 根据上文推倒的公式，计算换仓比例
        if len_new == 0:
            return 1
        res = 1 - intersection / max(len_old, len_new)
        return res

    hold_df['开仓调仓比例'] = hold_df.apply(lambda rows: cal_change_pct(rows, 'symbol_上周期', 'symbol'), axis=1)
    hold_df['平仓调仓比例'] = hold_df.apply(lambda rows: cal_change_pct(rows, 'symbol', 'symbol_下周期'), axis=1)

    '''
    计算完成后，hold_df 数据内容如下
          candle_begin_time  offset  方向  开仓调仓比例  平仓调仓比例
    2021-01-01 00:00:00       0  -1     1.0     0.0
    2021-01-01 03:00:00       0  -1     0.0     0.0
    2021-01-01 06:00:00       0  -1     0.0     0.0
    2021-01-01 09:00:00       0  -1     0.0     0.0
    2021-01-01 12:00:00       0  -1     0.0     0.0
    2021-01-01 15:00:00       0  -1     0.0     0.1
    2021-01-01 18:00:00       0  -1     0.1     0.0
    '''

    # 合并数据，这里继续做一个替换，整体调仓10%，和每个币调仓10%的效果是一样的。
    data = pd.merge(data, hold_df[['candle_begin_time', '开仓调仓比例', '平仓调仓比例']], 'left', on=['candle_begin_time'])
    # 计算仓位价值（带杠杆的情况下，不能用净值直接计算手续费和资金费）
    # data['仓位价值'] = data['每小时涨跌幅'].apply(lambda x: np.cumprod(np.array(x) + 1) * lvg)
    data['仓位价值'] = data.apply(lambda row: np.cumprod(np.array(row['每小时涨跌幅']) + 1) * row['lvg'], axis=1)
    data['仓位价值_byclose'] = data.apply(lambda row: np.cumprod(np.array(row['每小时涨跌幅_byclose']) + 1) * row['lvg'], axis=1)

    # 计算资金费
    def cal_fund_rate(row):
        """
        在资金曲线上加上资金费率的计算
        每个点应扣的资金费 =  ∑（资金费 x 方向 x 仓位价值）
        :param row: 每一行的数据
        :return:每个点应扣的资金费
        """
        pos_value = np.array(row['仓位价值'])
        # 每个点的资金费 = 资金费 x 方向 x 仓位价值
        fund_rate = np.array(row['fundingRate']) * row['方向']
        fund_fee = pos_value * fund_rate
        # 每个点应扣的资金费 = 历史的累计求和
        fund_fee = np.cumsum(fund_fee)
        return fund_fee

    # 计算每小时需要扣除的资金费（累计的数据，算在每个小时上）
    data['每小时资金费'] = data.apply(lambda rows: cal_fund_rate(rows), axis=1)

    # 如果选到下市的币种，会有周期长度对不上的情况，所以要做一下数据检查
    data['交易长度'] = data['每小时涨跌幅'].apply(len)
    data['最大周期长度'] = data.groupby('candle_begin_time')['交易长度'].transform('max')  # 选一个币的时候，存在单币周期不全的问题
    data['周期长度'] = int(pd.to_timedelta(hold_period) / pd.to_timedelta('1H'))  # 资金曲线小时级别，所有除以1h
    data.loc[data['candle_begin_time'] == data['candle_begin_time'].max(), '周期长度'] = data['最大周期长度']  # 修复处理最后一个周期数据长度不全的问题
    data['diff'] = data['周期长度'] - data['交易长度']  # 计算币种缺失的长度。0：表示没有数据缺失，持仓期间币种没有下架， 10：表示币种缺失10小时数据

    # 检查数据长度
    def check_length(row, col, method='last'):
        """
        有些币会在持仓过程中下架了，这里需要弥补一下数据长度
        :param row: 每一行数据
        :param col: 需要补充的数据长度
        :param method: 补充方式 last: 取最后一个数据进行向后补充  fill0: 使用0进行向后补充
        :return:
        """
        if row['diff'] == 0:
            return row[col]
        if method == 'fill0':
            return np.array(list(row[col]) + [0] * row['diff'])
        if method == 'last':
            return np.array(list(row[col]) + [row[col][-1]] * row['diff'])

    # 如果数据差异最大值超过0，表示需要进行数据检查
    if data['diff'].max() > 0:
        data['仓位价值'] = data.apply(lambda rows: check_length(rows, '仓位价值'), axis=1)
        data['每小时资金费'] = data.apply(lambda rows: check_length(rows, '每小时资金费'), axis=1)

        data['仓位价值_byclose'] = data.apply(lambda rows: check_length(rows, '仓位价值_byclose'), axis=1)

    # 计算手续费
    def cal_exchange_fee(row):
        """
        在资金曲线上加上手续费的计算
        每个点应扣的手续费 =  ∑（手续费率 x 仓位价值）
        :param row: 每一行的数据
        :return:每个点应扣的手续费
        """
        pos_value = np.array(row['仓位价值'])
        # 开仓的手续费 = lvg * rate * 开仓调整比例
        open_fee = row['lvg'] * row['rate'] * row['开仓调仓比例']
        # open_fee = lvg * row['rate'] * row['开仓调仓比例']
        # 平仓的手续费 = 仓位价值 * rate * 平仓调整比例
        close_fee = row['仓位价值'][-1] * row['rate'] * row['平仓调仓比例']

        # 小时级别的持仓长度（1H持仓长度为1，3H持仓长度为3）
        length = len(pos_value)
        if length > 1:
            # 如果周期长度大于2，每个点的手续费为：[开仓手续费，0,0,0,0，……，平仓手续费]（中间的0也可以是0个0）
            exchange_fee = [open_fee] + [0] * (length - 2) + [close_fee]
        else:
            # 如果周期长度等于1，即这根K线又要开仓又要平仓，手续费 = 开仓手续费 + 平仓手续费
            exchange_fee = [open_fee + close_fee]
        # 对手续费求和
        exchange_fee = np.cumsum(np.array(exchange_fee))
        return exchange_fee

    data['每小时手续费'] = data.apply(lambda rows: cal_exchange_fee(rows), axis=1)

    # 判断一下当前的方向
    data['每小时资金曲线'] = data.apply(lambda row: row['仓位价值'] - row['lvg'] + 1 if row['方向'] == 1 else row['lvg'] - row['仓位价值'] + 1, axis=1)
    data['每小时资金曲线_byclose'] = data.apply(lambda row: row['仓位价值_byclose'] - row['lvg'] + 1 if row['方向'] == 1 else row['lvg'] - row['仓位价值_byclose'] + 1, axis=1)
    # if data['方向'].iloc[0] == 1:
    #     # data['每小时资金曲线'] = data['仓位价值'].apply(lambda x: x - lvg + 1)
    #     data['每小时资金曲线'] = data.apply(lambda row: row['仓位价值'] - row['lvg'] + 1, axis=1)
    # else:
    #     # data['每小时资金曲线'] = data['仓位价值'].apply(lambda x: lvg - x + 1)
    #     data['每小时资金曲线'] = data.apply(lambda row: row['lvg'] - row['仓位价值'] + 1, axis=1)

    # 计算每小时资金曲线
    data['每小时资金曲线'] = data.apply(lambda rows: rows['每小时资金曲线'] - rows['每小时手续费'] - rows['每小时资金费'], axis=1)
    # 删除不必要的列
    data.drop(columns=['仓位价值', '每小时资金费', '每小时手续费'], inplace=True)
    # 重置索引
    data.reset_index(inplace=True)

    # 进行数据合并，将当周期上所有币种的信息进行合并处理
    groups = data.groupby(['candle_begin_time'])
    res_df = pd.DataFrame(groups['选币'].sum())
    res_df['方向'] = groups['方向'].last()
    res_df['offset'] = groups['offset'].last()
    res_df['每小时资金曲线'] = groups['每小时资金曲线'].apply(lambda x: np.array(x).mean(axis=0))
    res_df['每小时资金曲线_byclose'] = groups['每小时资金曲线_byclose'].apply(lambda x: np.array(x).mean(axis=0))
    res_df['调仓比例'] = groups['开仓调仓比例'].mean()
    res_df.reset_index(inplace=True)

    # 合并完整的时间周期，用于填充空缺的资金曲线
    res_df = pd.merge(left=trading_time_list, right=res_df, on='candle_begin_time', how='left')
    res_df['选币'].fillna(value='空仓', inplace=True)
    res_df['方向'].fillna(method='ffill', inplace=True)
    res_df['offset'].fillna(method='ffill', inplace=True)
    res_df['调仓比例'].fillna(value=0, inplace=True)
    hours = int(hold_period / pd.to_timedelta('1h'))
    res_df['每小时资金曲线'] = res_df.apply(lambda row: np.ones(hours) if row['选币'] == '空仓' else row['每小时资金曲线'], axis=1)
    res_df['每小时资金曲线_byclose'] = res_df.apply(lambda row: np.ones(hours) if row['选币'] == '空仓' else row['每小时资金曲线_byclose'], axis=1)

    return res_df


def create_trading_time(benchmark, hold_period, offset, start_date, end_date):
    """
    根据配置信息，构建当前offset下完整的交易时间，用于后续填充空缺数据
    :param benchmark: 基本数据
    :param hold_period: 持仓周期
    :param offset: 当前offset
    :param start_date: 回测开始时间
    :param end_date: 回测结束时间
    :return:
    """
    trading_time_list = benchmark.copy()
    trading_time_list.set_index('candle_begin_time', inplace=True)
    trading_time_list = trading_time_list.resample(rule=hold_period, base=offset).last().reset_index()
    trading_time_list = trading_time_list[trading_time_list['candle_begin_time'] >= pd.to_datetime(start_date)]
    trading_time_list = trading_time_list[trading_time_list['candle_begin_time'] < pd.to_datetime(end_date)]

    return trading_time_list


def calc_swap_pos_net_value(select_coin, leverage, hold_period, margin_rate, trading_time_list):
    """
    计算合约仓位净值信息
    :param select_coin: 所有的选币数据
    :param leverage: 杠杆
    :param hold_period: 持仓周期
    :param margin_rate: 保证金率
    :param trading_time_list: 完整的交易时间
    :return:
    """
    # ===计算当周期每个币的持仓权重
    select_coin['单方向选币数'] = select_coin.groupby(['candle_begin_time', '方向'])['symbol'].transform('size')
    select_coin['持仓权重'] = 1 / select_coin['单方向选币数']  # 假设每个方向都用1块钱去下单。多空加一起一共是2块

    # ===稍选合约的仓位
    swap_pos = select_coin[select_coin['symbol_type'] == 'swap']
    swap_pos['合约仓位价值'] = swap_pos.groupby('candle_begin_time')['持仓权重'].transform('sum')  # 将所有合约持仓权重合并
    swap_pos['现货仓位价值'] = 2 - swap_pos['合约仓位价值']  # 一共是2块，这里获取现货的持仓价值
    """
    公式推导过程：
    假设多空都是用1块钱去下单
        现货      持仓价值                合约      持仓价值
        BTC         1/5                 ADA         1/4
        ETH         1/5                 TRB         1/4
        BNB         1/5                 HNT         1/4
        LUNA        1/5                 AXS         1/4
        DOGE        1/5

    举例：
        条件：假设现货中LUNA，DOGE开合约，BTC，ETH，BNB购买现货
        
        当前仓位价值如下：
            现货价值  3/5
            合约价值  1+2/5=7/5
        
        条件：此时，我们开 1.5(3/2) 倍杠杆
        
        杠杆后仓位价值如下：
            现货价值  3/5 * 3/2 = 9/10
            合约价值  7/5 * 3/2 = 21/10
        
        现货的 9/10 此时是从原来2块中扣除，合约的实际保证金为： 2 - 9/10 = 11/10
        所以，合约仓位的实际杠杆为： 21/10 / (11/10) = 21/11 约等于 1.9 倍
        
        综上，
             合约实际杠杆率 =  合约仓位 * 杠杆 / (2 - 现货仓位 * 杠杆)
        
        通过公式，我们也能看出，杠杆最大不能超过2倍。
        主要原因是，你2块全买入了现货，合约没有保证金无法开仓
    """
    swap_pos['合约杠杆率'] = swap_pos['合约仓位价值'] * leverage / (2 - swap_pos['现货仓位价值'] * leverage)
    swap_pos['lvg'] = swap_pos['合约杠杆率']
    # ===计算合约的仓位净值
    swap_net_value = cal_net_value(swap_pos, hold_period, trading_time_list)
    if swap_net_value.empty:
        return pd.DataFrame()
    # ===计算是否爆仓
    swap_net_value['是否爆仓'] = swap_net_value['每小时资金曲线'].apply(lambda x: 1 if len(np.where(x < margin_rate)[0]) > 0 else np.nan)
    swap_net_value['是否爆仓'].fillna(method='ffill', inplace=True)
    # 判断是否存在爆仓信息
    if 1 in swap_net_value['是否爆仓'].to_list():
        print('杠杆开高了，嘣～～～沙卡拉卡～～～仓位没了')
        # =获取爆仓的索引
        first_index = swap_net_value[swap_net_value['是否爆仓'] == 1].index[0]
        other_index = swap_net_value[swap_net_value['是否爆仓'] == 1].index[1:-1]
        last_index = swap_net_value[swap_net_value['是否爆仓'] == 1].index[-1]

        """         
            时间         是否爆仓
        2021-01-01          0
        2021-01-02          1
        2021-01-03          1
        2021-01-04          1
        
        这里从 2021-01-03开始，将后面所有的每小时资金曲线设置为0
        """
        # =对非首个爆仓的资金曲线全部设置为空的资金曲线
        hours = int(hold_period / pd.to_timedelta('1h'))
        swap_net_value['每小时资金曲线'] = swap_net_value.apply(lambda row: np.zeros(hours) if row.name in other_index else row['每小时资金曲线'], axis=1)
        swap_net_value.loc[last_index, '每小时资金曲线'][0:] = 0
        """
        2021-01-02          1
        这里对 2021-01-02 当周期的资金曲线进行详细处理
        
        每小时资金曲线如下: [0.9, 0.8, 0.7, 0.06, 0.02, 0.01, 0.09]
        这里从第5个数据开始，净值低于保证金率的设置，所有后续的资金曲线需要全部设置为0
        先获取首次出现低于保证金率的数据下标，通过np.argwhere
        然后修改原数据
        """
        # =将首次出现爆仓的小时资金曲线单独处理
        _t = swap_net_value.loc[first_index, '每小时资金曲线']  # 获取首次出现爆仓的数据
        _t[np.argwhere(_t < margin_rate)[0][0]:] = 0  # 对首次出现爆仓之后的资金曲线全部设置为0
        swap_net_value.at[first_index, '每小时资金曲线'] = _t  # 修改数据之后，赋值回去

    swap_net_value.set_index('candle_begin_time', inplace=True)

    return swap_net_value


def save_factor(df, factor_class_list, root_path, symbol_type, symbol, factor_period='H'):
    """
    保存因子文件
    :param df: 含有因子数据的df
    :param factor_class_list: 参与计算的因子列表
    :param root_path: 项目目录位置
    :param symbol_type: 币种类型，spot/swap
    :param symbol: 币种名称
    :param factor_period: 因子周期
    :return:
    """
    for factor in factor_class_list:
        # 获取因子信息
        cols = [_ for _ in df.columns if factor == _.split('_')[0]]  # 解析因子列名。如果因子名称带下划线，将会解析出错
        if cols:
            # 构建存储目录路径
            save_file_path = os.path.join(root_path, f'data/数据整理/{symbol_type}/{symbol}/factors')
            # 目录不存，就创建目录
            if not os.path.exists(save_file_path):
                os.makedirs(save_file_path)  # 创建目录
            # 构建存储文件目录
            save_file_path = os.path.join(save_file_path, f'{symbol}_{factor}_{factor_period}.pkl')
            # 文件不存在，则保存。避免重复计算
            # if not os.path.exists(save_file_path):
            df[['candle_begin_time'] + cols].to_feather(save_file_path)


def read_factor(base_file, factor_dict, factor_period, hold_period, all_factor_list):
    """
    读取单个文件
    :param base_file: 存放周期数据路径
    :param factor_dict: 需要读取因子信息
    :param factor_period: 因子周期
    :return:
    """
    df = pd.read_feather(base_file)
    if df.empty:
        return pd.DataFrame()

    # feather存储的一个bug，list存在的结果会被转换程ndarray
    for column in df.columns:
        if isinstance(df[column][0], np.ndarray):
            df[column] = df[column].apply(lambda x: x.tolist())

    # 使用os.path.split()将路径拆分成目录和文件名
    directory, filename = os.path.split(base_file)
    # 使用os.path.dirname()获取目录的父目录
    parent_directory = os.path.dirname(directory)
    # 使用os.path.basename()获取目录的名称
    directory_name = os.path.basename(parent_directory)

    # ===数据整理
    from program.Config import black_list, white_list, start_date, end_date
    if black_list and directory_name+'-USDT' in black_list:
        return pd.DataFrame()
    if white_list and directory_name+'-USDT' not in white_list:
        return pd.DataFrame()
    df = df[df['是否交易'] == 1]  # 该周期不交易的币种
    df.dropna(subset=['下个周期_avg_price'], inplace=True)  # 最后几行数据，下个周期_avg_price为空
    # =只保留回测区间内的数据
    df = df[df['candle_begin_time'] >= pd.to_datetime(start_date)]
    df = df[df['candle_begin_time'] <= pd.to_datetime(end_date)]
    if df.empty:
        return pd.DataFrame()

   # 保存以close计算的每小时涨跌幅用来轮动
    df['每小时涨跌幅_byclose'] = df['每小时涨跌幅']

    # ===计算一些回测中需要用到的数据
    # 计算当周期的收益率
    df['ret_next'] = df['下个周期_avg_price'] / df['avg_price'] - 1
    # =将涨跌幅的格式改成list
    # df['开盘买入涨跌幅'] = df['开盘买入涨跌幅'].transform(lambda x: [x])  # 将开盘买入涨跌幅变成list格式
    # df['开盘卖出涨跌幅'] = df['开盘卖出涨跌幅'].transform(lambda x: [x])  # 将开盘卖出涨跌幅变成list格式
    # df['每小时涨跌幅'] = df['每小时涨跌幅'].transform(lambda x: x[1:])  # 去掉每小时涨跌幅的第一个涨跌幅数据
    # df['每小时涨跌幅'] = df['开盘买入涨跌幅'] + df['每小时涨跌幅']  # 用开盘买入涨跌幅补全第一个小时的涨跌幅数据
    # df['每小时涨跌幅'] = df['每小时涨跌幅'].transform(lambda x: x[:-1]) + df['开盘卖出涨跌幅']  # 用开盘卖出涨跌幅替换最后一个小时的涨跌幅数据
    # =如果持仓周期是1H，需要特殊处理，其涨跌幅就是ret_next
    if hold_period == '1H':
        # df['每小时涨跌幅'] = df['ret_next'].transform(lambda x: np.array([x]))
        df['每小时涨跌幅'] = df['ret_next'].transform(lambda x: [x])
    else:
        # df['每小时涨跌幅'] = df.apply(lambda row: np.concatenate(([row['开盘买入涨跌幅']], row['每小时涨跌幅'][1:-1], [row['开盘卖出涨跌幅']])), axis=1)
        df['每小时涨跌幅'] = df.apply(lambda row: [row['开盘买入涨跌幅']] + row['每小时涨跌幅'][1:-1] + [row['开盘卖出涨跌幅']], axis=1)

    # 获取这个目录下的因子信息
    for factor, factor_col in factor_dict.items():
        factor_path = os.path.join(parent_directory, f'factors/{directory_name}_{factor}_{factor_period}.pkl')
        factor_df = pd.read_feather(factor_path)

        # merge
        df = pd.merge(df, factor_df[['candle_begin_time'] + factor_col], 'left', 'candle_begin_time')

    # ===整理数据
    # 只保留指定字段，减少计算时间
    df = df[['candle_begin_time', 'offset', 'symbol', 'symbol_type', 'tag', 'ret_next', '每小时涨跌幅', '每小时涨跌幅_byclose',
             'fundingRate'] + all_factor_list]
    # 去除掉异常数据
    df[all_factor_list] = df[all_factor_list].replace([np.inf, -np.inf], np.nan)

    return df


def read_coin(root_path, hold_period, all_factor_list, if_use_spot, n_jobs, offset):
    """
    根据信息加载币种信息
    :param root_path: 项目根目录
    :param hold_period: 持仓周期
    :param all_factor_list: 加载的因子信息
    :param if_use_spot: 是否使用现货
    :param n_jobs: 多进程核心数控制
    :param offset: offset
    :return:
    """
    if offset not in list(range(0, int(hold_period[:-1]))):
        print(f'当前offset:【{offset}】，不属于hold_period：【{hold_period}】中，正确配置参数有：{list(range(0, int(hold_period[:-1])))}')
        exit()
    # 解析因子周期
    factor_period = hold_period[-1]

    # 现货模式，需要加载现货与合约数据
    if if_use_spot:
        period_path = os.path.join(root_path, f'data/数据整理/*/*/periods')
    else:  # 合约模式，只需要加载合约数据
        period_path = os.path.join(root_path, f'data/数据整理/swap/*/periods')

    # 获取存放周期目录下的的指定周期文件路径
    base_file_path = glob(period_path + f'/*_{hold_period}_{offset}.pkl')

    # 创建一个空字典用于存放结果
    factor_dict = {}

    # 遍历列表中的每个元素
    for item in all_factor_list:
        # 使用下划线分割字符串
        parts = item.split('_')
        if len(parts) > 2:
            print('因子名称中含有下划线，跳过改因子:', item)
            continue

        # 第一部分作为键，第二部分作为值
        key = parts[0]
        value = item

        # 如果键已经存在于字典中，将值添加到对应键的列表
        if key in factor_dict:
            factor_dict[key].append(value)
        else:
            # 如果键不存在，创建一个新的键值对
            factor_dict[key] = [value]

    # 不存在配置的因子，跳过检查
    if not factor_dict:
        return pd.DataFrame()

    # 多进程读取数据
    all_df = Parallel(n_jobs=n_jobs)(
        delayed(read_factor)(base_file, factor_dict, factor_period, hold_period, all_factor_list)
        for base_file in base_file_path
    )

    if not all_df:
        return pd.DataFrame()

    # 数据合并
    all_df = pd.concat(all_df, ignore_index=True)
    # 按时间和币种名对数据进行重新排序并重置索引
    all_df.sort_values(by=['candle_begin_time', 'symbol'], inplace=True)
    all_df.reset_index(drop=True, inplace=True)

    return all_df


def factor_info_to_str(f_infos):
    """
    因子信息转成字符串
    :param f_infos:
    :return:
    """
    infos = ''
    for f in f_infos.keys():
        info = f + str(f_infos[f]) + '+'
        infos += info
    infos = infos[:-1]
    return infos


def str_to_factor_info(info_str):
    """
    字符串转成因子信息
    :param info_str:
    :return:
    """
    factor_dict = {}
    infos = info_str.split('+')
    for info in infos:
        if 'True' in info:
            factor_dict[info[:-4]] = True
        else:
            factor_dict[info[:-5]] = False
    return factor_dict


def _warning(msg):
    """
    警告信息
    :param msg: 信息的内容
    """
    print('*' * 40)
    print('*' * 40)
    print()
    print('警告!!!!!!!')
    print(msg)
    print()
    print('*' * 40)
    print('*' * 40)


def check_leverage(strategy, if_use_spot, leverage):
    """
    检查杠杆
    :param strategy: 策略
    :param if_use_spot: 是否使用现货
    :param leverage: 杠杆
    """
    if if_use_spot:
        if leverage >= 2:
            _warning('现货模式下，杠杆 >= 2 回测结果可能会出现问题')
        if strategy.long_select_coin_num == 0:
            _warning('现货模式下，无法配置纯空头模式，程序退出')
            exit()
        if strategy.short_select_coin_num == 0 and leverage > 1:
            _warning('现货模式下，纯多头模式杠杆无法超过 1，程序退出')
            exit()
    else:
        if strategy.long_select_coin_num == 0 and isinstance(strategy.short_select_coin_num, str):
            _warning('合约模式下，无法配置【long_nums】，程序退出')
            exit()


def revise_data_length(data, data_len, value=0):
    """
    校正数据长度
    原数据过长，则进行切片
    原数据果断，则使用0填充
    :param data: 原数据
    :param data_len: 资金曲线的数据长度
    :param value: 填充的数据
    :return: 校正后的数据
    """
    if len(data) >= data_len:
        data = data[0:data_len]
    elif len(data) < data_len:
        data.extend([value] * (data_len - len(data)))

    return data
