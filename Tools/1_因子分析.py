# -*- coding: utf-8 -*-
"""
中性策略框架 | 邢不行 | 2024分享会
author: 邢不行
微信: xbx6660
"""
from datetime import datetime

import pandas as pd
from Tools.utils.PlotFunctions import *
from program.Config import *
from Tools.utils.tFunctions import *
import warnings

warnings.filterwarnings('ignore')

# =====需要配置的东西=====
factor = 'QuoteVolumeMean_7'  # 你想测试的因子
period = '1D'  # 配置读入数据的period
target = 'ret_next'  # 测试因子与下周期涨跌幅的IC，可以选择其他指标比如夏普率等
need_shift = False  # target这列需不需要shift，如果为True则将target这列向下移动一个周期
is_use_spot = False  # 是否使用现货数据进行因子测试
multiple_process = False  # True为并行，False为串行
offset_list = list(range(0, int(period[:-1])))  # 根据配置的period获取offset
# 如果target列向下shift1个周期，则更新下target指定的列
if need_shift:
    target = '下周期_' + target
# =====需要配置的东西=====

# =====几乎不需要配置的东西=====
bins = 10  # 分箱数
limit = 20  # 1.某个周期至少有20个币，否则过滤掉这个周期；注意：limit需要大于bins；可能会造成不同因子开始时间不一致
next_ret = '每小时涨跌幅'  # 使用下周期每天涨跌幅画分组持仓走势图
data_folder = root_path + f'/data/数据整理/'  # 配置读入数据的文件夹路径
b_rate = 4 / 10000  # 买入手续费
s_rate = 4 / 10000  # 卖出手续费
# 创建列表，用来保存各个offset的数据
IC_list = []  # IC数据列表
group_nv_list = []  # 分组净值列表
group_hold_value_list = []  # 分组持仓走势列表
style_corr_list = []  # 市值分析数据列表
style_factor_param = 7  # 风格因子的参数
# =====几乎不需要配置的东西=====


def factor_analysis(offset):
    # 构建风格因子
    factor_list = get_file_in_folder(root_path + '/program/factors/', file_type='.py', filters=['__init__'],
                                     drop_type=True)
    style_factor_list = [_ for _ in factor_list if '风格' in _]
    all_factor_list = [factor]
    for _factor in style_factor_list:
        all_factor_list.append(f'{_factor}_{style_factor_param}')
    # 读入数据
    df = get_factor_by_period(root_path, period, is_use_spot, target, need_shift, start_date, end_date, all_factor_list,
                              offset, n_jobs)
    # 如果返回的数据为空，则跳过该offset继续读取下一个offset的数据
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 删除必要字段为空的部分
    df = df.dropna(subset=[factor, target, next_ret], how='any')
    # ===根据limit，保留周期的币种数量大于limit的周期
    # =拿到每个周期的币种数量
    stock_nums = df.groupby('candle_begin_time').size()
    # =保留每个周期的币种数量大于limit的日期
    save_dates = stock_nums[stock_nums > limit].index
    # =如果交易日期save_dates中，则是否保留列为True
    df['是否保留'] = df['candle_begin_time'].map(lambda x: x in save_dates)
    # =取出 是否保留==True 的数据
    df = df[df['是否保留'] == True].reset_index(drop=True)

    # 将数据按照交易日期和offset进行分组
    df = offset_grouping(df, factor, bins)
    # ===计算这个offset下的IC
    IC = get_IC(df, factor, target, offset)
    # ===计算这个offset下的分组资金曲线、分组持仓走势
    group_nv, group_hold_value = get_group_nv(df, next_ret, b_rate, s_rate, offset)
    # ===计算风格暴露
    style_corr = get_style_corr(df, factor, offset)
    return IC, group_nv, group_hold_value, style_corr


print(f'开始进行 {factor} 因子分析...')
s_date = datetime.datetime.now()  # 记录开始时间
if multiple_process:
    result_list = Parallel(n_jobs=n_jobs)(delayed(factor_analysis)(offset) for offset in offset_list)
    # 将返回的数据添加到对应的列表中
    for idx, offset in enumerate(offset_list):
        if not result_list[idx][0].empty:
            IC_list.append(result_list[idx][0])
            group_nv_list.append(result_list[idx][1])
            group_hold_value_list.append(result_list[idx][2])
            style_corr_list.append(result_list[idx][3])

else:
    for offset in offset_list:  # 遍历offset进行因子分析
        IC, group_nv, group_hold_value, style_corr = factor_analysis(offset)
        if not IC.empty:
            # 将返回的数据添加到对应的列表中
            IC_list.append(IC)
            group_nv_list.append(group_nv)
            group_hold_value_list.append(group_hold_value)
            style_corr_list.append(style_corr)

# 生成一个包含图的列表，之后的代码每画出一个图都添加到该列表中，最后一起画出图
fig_list = []
print('正在汇总各offset数据并画图...')
start_date = datetime.datetime.now()  # 记录开始时间
# ===计算IC、累计IC以及IC的评价指标
IC, IC_info = IC_analysis(IC_list)
# =画IC走势图，并将IC图加入到fig_list中，最后一起画图
Rank_fig = draw_ic_plotly(x=IC['candle_begin_time'], y1=IC['RankIC'], y2=IC['累计RankIC'], title='因子RankIC图',
                          info=IC_info)
fig_list.append(Rank_fig)
# =画IC热力图（年份月份），并将图添加到fig_list中
# 处理IC数据，生成每月的平均IC
IC_month = get_IC_month(IC)
# 画图并添加
hot_fig = draw_hot_plotly(x=IC_month.columns, y=IC_month.index, z=IC_month, title='RankIC热力图(行：年份，列：月份)')
fig_list.append(hot_fig)

# ===计算分组资金曲线、分箱图、分组持仓走势
group_curve, group_value, group_hold_value = group_analysis(group_nv_list, group_hold_value_list)
# =画分组资金曲线...
cols_list = [col for col in group_curve.columns if '第' in col]
group_fig = draw_line_plotly(x=group_curve['candle_begin_time'], y1=group_curve[cols_list], y2=group_curve['多空净值'],
                             if_log=True, title='分组资金曲线')
fig_list.append(group_fig)

# =画分箱净值图
group_fig = draw_bar_plotly(x=group_value['分组'], y=group_value['净值'], title='分组净值')
fig_list.append(group_fig)
# =画分组持仓走势
group_fig = draw_line_plotly(x=group_hold_value['时间'], y1=group_hold_value[cols_list], update_xticks=True,
                             if_log=False, title='分组持仓走势')
fig_list.append(group_fig)

# ===计算风格暴露
style_corr = style_analysis(style_corr_list)
if not style_corr.empty:
    # =画风格暴露图
    style_fig = draw_bar_plotly(x=style_corr['风格'], y=style_corr['相关系数'], title='因子风格暴露图')
    fig_list.append(style_fig)

# ===整合上面所有的图
merge_html(root_path, fig_list=fig_list, strategy_file=f'{factor}因子分析报告')
print(f'汇总数据并画图完成，耗时：{datetime.datetime.now() - start_date}')
print(f'{factor} 因子分析完成，耗时：{datetime.datetime.now() - s_date}')
