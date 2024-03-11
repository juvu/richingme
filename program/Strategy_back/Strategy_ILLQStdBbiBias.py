# -*- coding: utf-8 -*-
"""
中性策略框架 | 邢不行 | 2024分享会
author: 邢不行
微信: xbx6660
"""
import os

# 取出策略文件名中_后的部分
stg_name = os.path.basename(__file__).split('.')[0].split('_')[1]

# 持仓周期。目前回测支持日线级别、小时级别。例：1H，6H，3D，7D......
# 当持仓周期为D时，选币指标也是按照每天一根K线进行计算。
# 当持仓周期为H时，选币指标也是按照每小时一根K线进行计算。
hold_period = '1H'.replace('h', 'H').replace('d', 'D')
# 配置offset
offset = 0
# 是否使用现货
if_use_spot = True  # True：使用现货。False：不使用现货，只使用合约。
if_pure_spot = False  # True：只买现货。False：如果有对应合约就买对应合约，节约手续费。

# 多头选币数量。1 表示做多一个币; 0.1 表示做多10%的币
long_select_coin_num = 0.1
# 空头选币数量。1 表示做空一个币; 0.1 表示做空10%的币
# short_select_coin_num = 0.1
short_select_coin_num = 'long_nums'  # long_nums意为着空头数量和多头数量保持一致。最多为所有合约的数量。注意：多头为0的时候，不能配置'long_nums'

# 多头的选币因子列名。
long_factor = '因子'  # 因子：表示使用复合因子，默认是 factor_list 里面的因子组合。需要修改 calc_factor 函数配合使用
# 空头的选币因子列名。多头和空头可以使用不同的选币因子
short_factor = '因子'

# 选币因子信息列表，用于`2_选币_单offset.py`，`3_计算多offset资金曲线.py`共用计算资金曲线
factor_list = [
    ('ILLQStdBbiBias', True, 168, 1),  # 因子名（和factors文件中相同），排序方式，参数，权重。
]

# 确认过滤因子及其参数，用于`2_选币_单offset.py`进行过滤
filter_list = [
    ('PctChange', 168),
]


def after_merge_index(df, symbol, factor_dict, data_dict):
    """
    合并指数数据之后的处理流程，非必要。
    本函数住要的作用如下：
            1、指定K线上不常用的数据在resample时的规则，例如：'taker_buy_quote_asset_volume': 'sum'
            2、合并外部数据，并指定外部数据在resample时的规则。例如：（伪代码）
                    chain_df = pd.read_csv(chain_path + symbol)  # 读取指定币种链上数据
                    df = pd.merge(df,chain_df,'left','candle_begin_time') # 将链上数据合并到日线上
                    factor_dict['gas_total'] = 'sum' # 链上的gas_total字段，在小时数据转日线数据时用sum处理
                    data_dict['gas_total'] = 'sum' # 链上的gas_total字段，在最后一次resample中的处理规则
    :param df:
    :param symbol:
    :param factor_dict: 小时级别resample到日线级别时使用(计算日线级别因子时需要，计算小时级别因子时不需要)
    :param data_dict: resample生成最终数据时使用
    :return:
    """

    # 专门处理转日线是的resample规则
    factor_dict['taker_buy_quote_asset_volume'] = 'sum'  # 计算日线级别因子前先resample到日线数据
    factor_dict['trade_num'] = 'sum'

    return df, factor_dict, data_dict


def after_resample(df, symbol):
    """
    数据重采样之后的处理流程，非必要
    :param df:          传入的数据
    :param symbol:      币种名称
    :return:
    """

    return df


# =====================以上是数据整理部分封装转的策略代码==========================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# =======================以下是选币函数封装的策略代码=============================


def calc_factor(df, **kwargs):
    """
    计算因子

    多空使用相同的两个因子

        1。多空相同的单因子选币，不需要编写代码

        2。多空相同的复合因子，本案例代码不需要修改，直接使用即可。《本案例代码就是多空相同的复合因子选币》

    多空使用不同的两个因子

        1。多空不同的单因子选币，不需要编写代码

        2。多空分离选币，需要根据具体因子具体改写
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            !!!!!这里改写需要一定的代码能力!!!!!
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            多空存在某一方是复合因子，另一方是单因子，则只需要计算一个复合因子即可。代码可以参考下方案例代码

            如果多空都是复合因子，多空需要分别是计算复合因子。

    :param df:          原数据
    :return:
    """
    # 接受外部回测的因子列表，这里主要是去适配`4_遍历选币参数.py`
    external_list = kwargs.get('external_list', [])
    if external_list:  # 如果存在外部回测因子列表，则使用外部因子列表
        _factor_list = external_list
    else:  # 如过不存在外部回测因子列表，默认使用当前策略的因子列表
        _factor_list = factor_list

    # 多空相同的复合因子计算
    if long_factor == short_factor == '因子':
        df[long_factor] = 0
        for factor_name, if_reverse, parameter_list, weight in _factor_list:
            col_name = f'{factor_name}_{str(parameter_list)}'
            # 计算单个因子的排名
            df[col_name + '_rank'] = df.groupby('candle_begin_time')[col_name].rank(ascending=if_reverse, method='min')
            # 将因子按照权重累加
            df[long_factor] += (df[col_name + '_rank'] * weight)

    return df


def before_filter(df, **kwargs):
    """
    前置过滤函数
    自定义过滤规则，可以对多空分别自定义过滤规则

    :param df:                  原始数据
    :return:                    过滤后的数据
    """
    # 接受外部回测的因子列表，这里主要是去适配`5_查看历年参数平原.py`
    ex_filter_list = kwargs.get('ex_filter_list', [])
    if ex_filter_list:  # 如果存在外部回测因子列表，则使用外部因子列表
        _filter_list = ex_filter_list
    else:  # 如过不存在外部回测因子列表，默认使用当前策略的因子列表
        _filter_list = filter_list

    df_long = df.copy()
    df_short = df.copy()

    # 如果过滤列表中只有一个因子
    if len(_filter_list) == 1:
        filter_factor = _filter_list[0][0] + '_' + str(_filter_list[0][1])
        # 配置了过滤因子信息，则进行过滤操作
        df_long['filter_rank'] = df_long.groupby('candle_begin_time')[filter_factor].rank(ascending=True, pct=True)
        df_long = df_long[(df_long['filter_rank'] < 0.8)]

        df_short['filter_rank'] = df_short.groupby('candle_begin_time')[filter_factor].rank(ascending=True, pct=True)
        df_short = df_short[(df_short['filter_rank'] < 0.8)]
    elif len(_filter_list) > 1:  # 如果使用多个因子进行过滤，在这里进行填写
        filter_factor = _filter_list[0][0] + '_' + str(_filter_list[0][1])
        # 配置了过滤因子信息，则进行过滤操作
        df_long['filter_rank'] = df_long.groupby('candle_begin_time')[filter_factor].rank(ascending=True, pct=True)
        df_long = df_long[(df_long['filter_rank'] < 0.8)]

        df_short['filter_rank'] = df_short.groupby('candle_begin_time')[filter_factor].rank(ascending=True, pct=True)
        df_short = df_short[(df_short['filter_rank'] < 0.8)]

        filter_factor = _filter_list[1][0] + '_' + str(_filter_list[1][1])
        df_long = df_long[(df_long[filter_factor] <= 0.1)]
        df_short = df_short[(df_short[filter_factor] <= 0.1)]

        filter_factor = 'Volatility_168'
        df_long['filter_rank'] = df_long.groupby('candle_begin_time')[filter_factor].rank(ascending=True, pct=True)
        df_long = df_long[(df_long['filter_rank'] > 0.2)]

        df_short['filter_rank'] = df_short.groupby('candle_begin_time')[filter_factor].rank(ascending=True, pct=True)
        df_short = df_short[(df_short['filter_rank'] > 0.2)]




    return df_long, df_short
