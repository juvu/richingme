# -*- coding: utf-8 -*-
"""
中性策略框架 | 邢不行 | 2024分享会
author: 邢不行
微信: xbx6660
"""
import numpy as np
import pandas as pd
from program.Functions import *
import math
import datetime


def float_num_process(num, return_type=float, keep=2, max=5):
    """
    针对绝对值小于1的数字进行特殊处理，保留非0的N位（N默认为2，即keep参数）
    输入  0.231  输出  0.23
    输入  0.0231  输出  0.023
    输入  0.00231  输出  0.0023
    如果前面max个都是0，直接返回0.0
    :param num: 输入的数据
    :param return_type: 返回的数据类型，默认是float
    :param keep: 需要保留的非零位数
    :param max: 最长保留多少位
    :return:
        返回一个float或str
    """

    # 如果输入的数据是0，直接返回0.0
    if num == 0.:
        return 0.0

    # 绝对值大于1的数直接保留对应的位数输出
    if abs(num) > 1:
        return round(num, keep)
    # 获取小数点后面有多少个0
    zero_count = -int(math.log10(abs(num)))
    # 实际需要保留的位数
    keep = min(zero_count + keep, max)

    # 如果指定return_type是float，则返回float类型的数据
    if return_type == float:
        return round(num, keep)
    # 如果指定return_type是str，则返回str类型的数据
    else:
        return str(round(num, keep))


def get_factor_by_period(root_path, period, is_use_spot, target, need_shift, start_date, end_date, all_factor_list, offset, n_jobs):
    """
    读取数据函数
    :param root_path: 项目根目录
    :param period: 配置的周期
    :param is_use_spot: 是否使用现货数据进行分析
    :param target: 目标列
    :param need_shift: 目标列是否需要shift
    :param start_date: 开始时间
    :param end_date: 结束时间
    :param offset: 读入的offset
    :param all_factor_list: 所有因子数据
    :param n_jobs: 多进程核心数控制
    :return:
        返回读取到的数据
    """
    df = read_coin(root_path, period, all_factor_list, is_use_spot, n_jobs, offset)
    if df.empty:
        print('读取文件为空，请检查配置!')
        exit()

    # target列是否需要shift
    if need_shift:
        df['下周期_' + target] = df.groupby('symbol').apply(lambda x: x[target].shift()).reset_index(0)[target]
    # 将异常数据替换为nan
    df = df.replace([np.inf, -np.inf], np.nan)

    # 判断是否使用现货数据进行分析
    if is_use_spot:
        df = df[df['symbol_type'] == 'spot'].reset_index(drop=True)
    else:
        df = df[df['symbol_type'] == 'swap'].reset_index(drop=True)

    # 筛选符合条件的offset
    df = df[df['offset'] == offset]
    return df


def offset_grouping(df, factor, bins):
    """
    分组函数
    :param df: 原数据
    :param factor: 因子名
    :param bins: 分组的数量
    :return:
        返回一个df数据，包含groups列
    """

    # 根据factor计算因子的排名
    df['因子_排名'] = df.groupby(['candle_begin_time'])[factor].rank(ascending=True, method='first')
    # 根据因子的排名进行分组
    df['groups'] = df.groupby(['candle_begin_time'])['因子_排名'].transform(
        lambda x: pd.qcut(x, q=bins, labels=range(1, bins + 1), duplicates='drop'))

    return df


def IC_analysis(IC_list):
    """
    整合各个offset的IC数据并计算相关的IC指标
    :param IC_list: 各个offset的IC数据
    :return:返回IC数据、IC字符串
    """

    print('正在汇总因子IC分析...')

    # 将各个offset的数据合并 并 整理
    IC = pd.concat(IC_list, axis=0)
    IC = IC.sort_values('candle_begin_time').reset_index(drop=True)

    # 计算累计RankIC；注意：因为我们考虑了每个offset，所以这边为了使得各个不同period之间的IC累计值能够比较，故除以offset的数量
    IC['累计RankIC'] = IC['RankIC'].cumsum() / (IC['offset'].max() + 1)

    # ===计算IC的统计值，并进行约等
    # =IC均值
    IC_mean = float_num_process(IC['RankIC'].mean())
    # =IC标准差
    IC_std = float_num_process(IC['RankIC'].std())
    # =ICIR
    ICIR = float_num_process(IC_mean / IC_std)
    # =IC胜率
    # 如果累计IC为正，则计算IC为正的比例
    if IC['累计RankIC'].iloc[-1] > 0:
        IC_ratio = str(float_num_process((IC['RankIC'] > 0).sum() / len(IC) * 100)) + '%'
    # 如果累计IC为负，则计算IC为负的比例
    else:
        IC_ratio = str(float_num_process((IC['RankIC'] < 0).sum() / len(IC) * 100)) + '%'

    # 将上述指标合成一个字符串，加入到IC图中
    IC_info = f'IC均值：{IC_mean}，IC标准差：{IC_std}，ICIR：{ICIR}，IC胜率：{IC_ratio}'

    return IC, IC_info


def get_corr_month(corr):
    """
    生成IC月历
    :param corr: IC数据
    :return:
        返回IC月历的df数据
    """

    print('正在进行IC月历计算...')

    # resample到月份数据
    corr['candle_begin_time'] = pd.to_datetime(corr['candle_begin_time'])
    corr.set_index('candle_begin_time', inplace=True)
    corr_month = corr.resample('M').agg({'RankIC': 'mean'})
    corr_month.reset_index(inplace=True)
    # 提取出年份和月份
    corr_month['年份'] = corr_month['candle_begin_time'].map(lambda x: str(x)[:4])
    corr_month['月份'] = corr_month['candle_begin_time'].map(lambda x: str(x)[5:7])
    # 将年份月份设置为index，在将月份unstack为列
    corr_month = corr_month.set_index(['年份', '月份'])['RankIC']
    corr_month = corr_month.unstack('月份')
    # 计算各月平均的IC
    corr_month.loc['各月平均', :] = corr_month.mean(axis=0)
    # 按年份大小排名
    corr_month = corr_month.sort_index(ascending=False)

    return corr_month


def group_analysis(group_nv_list, group_hold_value_list):
    """
    针对分组数据进行分析，给出分组的资金曲线、分箱图以及各分组的未来资金曲线
    :param group_nv_list: 各个offset的分组净值数据
    :param group_hold_value_list: 各个offset的分组持仓走势数据
    :return:
        返回分组资金曲线、分箱图、分组持仓走势数据
    """

    print('正在汇总因子分组分析...')

    # 生成时间轴
    dates = []
    for group_nv in group_nv_list:
        dates.extend(list(set(group_nv['candle_begin_time'].to_list())))
    time_df = pd.DataFrame(sorted(dates), columns=['candle_begin_time'])

    # 遍历各个offset的资金曲线数据，合并到时间轴上，将合并后的数据append到列表中
    nv_list = []
    for group_nv in group_nv_list:
        group_nv = group_nv.groupby('groups').apply(
            lambda x: pd.merge(time_df, x, 'left', 'candle_begin_time').fillna(method='ffill'))
        group_nv.reset_index(drop=True, inplace=True)
        nv_list.append(group_nv)

    # 将所有offset的分组资金曲线数据合并
    nv_df = pd.concat(nv_list, ignore_index=True)
    # 计算当前数据有多少个分组
    bins = nv_df['groups'].max()
    # 计算不同offset的每个分组的平均净值
    group_curve = nv_df.groupby(['candle_begin_time', 'groups'])['净值'].mean().reset_index()
    # 将数据按照展开
    group_curve = group_curve.set_index(['candle_begin_time', 'groups']).unstack().reset_index()
    # 重命名数据列
    group_cols = ['candle_begin_time'] + [f'第{i}组' for i in range(1, bins + 1)]
    group_curve.columns = group_cols

    # 计算多空净值走势
    # 获取第一组的涨跌幅数据
    first_group_ret = group_curve['第1组'].pct_change()
    first_group_ret.fillna(value=group_curve['第1组'].iloc[0] - 1, inplace=True)
    # 获取最后一组的涨跌幅数据
    last_group_ret = group_curve[f'第{bins}组'].pct_change()
    last_group_ret.fillna(value=group_curve[f'第{bins}组'].iloc[0] - 1, inplace=True)
    # 判断到底是多第一组空最后一组，还是多最后一组空第一组
    if group_curve['第1组'].iloc[-1] > group_curve[f'第{bins}组'].iloc[-1]:
        ls_ret = (first_group_ret - last_group_ret) / 2
    else:
        ls_ret = (last_group_ret - first_group_ret) / 2
    # 计算多空净值曲线
    group_curve['多空净值'] = (ls_ret + 1).cumprod()
    # 计算绘制分箱所需要的数据
    group_value = group_curve[-1:].T[1:].reset_index()
    group_value.columns = ['分组', '净值']

    # 合并各个offset的持仓走势数据
    all_group_hold_value = pd.concat(group_hold_value_list, axis=0)
    # 取出需要求各个offset平均的列
    mean_cols = [col for col in all_group_hold_value.columns if '第' in col]
    # 新建空df
    group_hold_value = pd.DataFrame()
    # 设定时间列
    group_hold_value['时间'] = all_group_hold_value['时间'].unique()
    # 求各个组的mean
    for col in mean_cols:
        group_hold_value[col] = all_group_hold_value.groupby('时间')[col].mean()

    return group_curve, group_value, group_hold_value


def style_analysis(style_corr_list):
    """
    计算因子的风格暴露
    :param style_corr_list: 各个offset的风格暴露数据
    :return:
       返回因子的风格暴露的数据
    """
    # 合并各个offset的风格暴露数据
    style_corr = pd.concat(style_corr_list, axis=0)
    if len(style_corr) < 1:
        return style_corr
    # 对各offset求平均
    style_corr = style_corr.groupby('风格')['相关系数'].mean().to_frame().reset_index()

    return style_corr


def industry_analysis(df, factor, target, industry_col, industry_name_change):
    """
    计算分行业的IC
    :param df: 原始数据
    :param factor: 因子列
    :param target: 目标列
    :param industry_col: 配置的行业列名
    :return:
        返回各个行业的RankIC数据、占比数据
    """

    print('正在进行因子行业分析...')

    def get_industry_data(temp):
        """
        计算分行业IC、占比
        :param temp: 每个行业的数据
        :return:
            返回IC序列的均值、第一组占比、最后一组占比
        """
        # 计算每个行业的IC值
        ic = temp.groupby('candle_begin_time').apply(lambda x: x[factor].corr(x[target], method='spearman'))
        # 计算每个行业的第一组的占比和最后一组的占比
        part_min_group = temp.groupby('candle_begin_time').apply(lambda x: (x['groups'] == min_group).sum())
        part_max_group = temp.groupby('candle_begin_time').apply(lambda x: (x['groups'] == max_group).sum())
        part_min_group = part_min_group / all_min_group
        part_max_group = part_max_group / all_max_group

        return [ic.mean(), part_min_group.mean(), part_max_group.mean()]

    # 替换行业名称
    df[industry_col] = df[industry_col].replace(industry_name_change)
    # 获取以因子分组第一组和最后一组的数量
    min_group, max_group = df['groups'].min(), df['groups'].max()
    all_min_group = df.groupby('candle_begin_time').apply(lambda x: (x['groups'] == min_group).sum())
    all_max_group = df.groupby('candle_begin_time').apply(lambda x: (x['groups'] == max_group).sum())
    # 以行业分组计算IC及占比，并处理数据
    industry_data = df.groupby(industry_col).apply(get_industry_data).to_frame().reset_index()
    # 取出IC数据、行业占比_第一组数据、行业占比_最后一组数据
    industry_data['RankIC'] = industry_data[0].map(lambda x: x[0])
    industry_data['行业占比_第一组'] = industry_data[0].map(lambda x: x[1])
    industry_data['行业占比_最后一组'] = industry_data[0].map(lambda x: x[2])
    # 处理数据
    industry_data.drop(0, axis=1, inplace=True)
    # 以IC排序
    industry_data.sort_values('RankIC', ascending=False, inplace=True)

    return industry_data


def market_value_analysis(df, factor, target, market_value, bins=10):
    """
    计算分市值的IC数据
    :param df: 原数据
    :param factor: 因子名
    :param target: 目标名
    :param market_value: 配置的市值列名
    :param bins: 分组的数量
    :return:
        返回各个市值分组的IC、占比数据
    """

    print('正在进行因子市值分析...')

    # 先对市值数据进行排名以及分组
    df['市值_排名'] = df.groupby(['candle_begin_time'])[market_value].rank(ascending=True, method='first')
    df['市值分组'] = df.groupby(['candle_begin_time'])['市值_排名'].transform(
        lambda x: pd.qcut(x, q=bins, labels=range(1, bins + 1), duplicates='drop'))

    def get_market_value_data(temp):
        '''
        计算分市值IC、占比
        :param temp: 每个市值分组的数据
        :return:
            返回IC序列的均值、第一组占比、最后一组占比
        '''
        # 计算每个市值分组的IC值
        ic = temp.groupby('candle_begin_time').apply(lambda x: x[factor].corr(x[target], method='spearman'))
        # 计算每个市值分组的第一组的占比和最后一组的占比
        part_min_group = temp.groupby('candle_begin_time').apply(lambda x: (x['groups'] == min_group).sum())
        part_max_group = temp.groupby('candle_begin_time').apply(lambda x: (x['groups'] == max_group).sum())
        part_min_group = part_min_group / all_min_group
        part_max_group = part_max_group / all_max_group

        return [ic.mean(), part_min_group.mean(), part_max_group.mean()]

    # 获取以因子分组第一组和最后一组的数量
    min_group, max_group = df['groups'].min(), df['groups'].max()
    all_min_group = df.groupby('candle_begin_time').apply(lambda x: (x['groups'] == min_group).sum())
    all_max_group = df.groupby('candle_begin_time').apply(lambda x: (x['groups'] == max_group).sum())
    # 根据市值分组计算IC及占比，并处理数据
    market_value_data = df.groupby('市值分组').apply(get_market_value_data).to_frame().reset_index()
    # 取出IC数据、市值占比_第一组数据、市值占比_最后一组数据
    market_value_data['RankIC'] = market_value_data[0].map(lambda x: x[0])
    market_value_data['市值占比_第一组'] = market_value_data[0].map(lambda x: x[1])
    market_value_data['市值占比_最后一组'] = market_value_data[0].map(lambda x: x[2])
    # 处理数据
    market_value_data.drop(0, axis=1, inplace=True)
    # 以市值分组大小排序
    market_value_data.sort_index(ascending=True, inplace=True)

    return market_value_data


def get_IC(df, factor, target, offset):
    """
    计算IC等一系列指标
    :param df: 数据
    :param factor: 因子列名：测试的因子名称
    :param target: 目标列名：计算IC时的下周期数据
    :param offset: 当前执行的是哪个offset的数据
    :return:
        返回计算得到的IC数据
    """

    print('正在进行因子IC分析...')
    start_date = datetime.datetime.now()  # 记录开始时间

    # 计算IC并处理数据
    corr = df.groupby('candle_begin_time').apply(lambda x: x[factor].corr(x[target], method='spearman')).to_frame()
    corr = corr.rename(columns={0: 'RankIC'}).reset_index()

    # 记录offset
    corr['offset'] = offset

    print(f'因子IC分析完成，耗时：{datetime.datetime.now() - start_date}')

    return corr


def get_group_nv(df, next_ret, b_rate, s_rate, offset):
    """
    针对分组数据进行分析，给出分组的资金曲线、分箱图以及各分组的未来资金曲线
    :param df: 输入的数据
    :param next_ret: 未来涨跌幅的list
    :param b_rate: 买入手续费用
    :param s_rate: 卖出手续费用
    :param offset: 当前执行的是哪个offset的数据
    :return:
        返回分组资金曲线、分组持仓走势数据
    """

    print('正在进行因子分组分析...')
    start_date = datetime.datetime.now()  # 记录开始时间

    # 由于会对原始的数据进行修正，所以需要把数据copy一份
    temp = df.copy()

    # 将持仓周期的众数当做标准的持仓周期数
    temp['持仓周期'] = temp[next_ret].apply(lambda x: len(x))
    hold_nums = int(temp['持仓周期'].mode())
    temp[next_ret] = temp[next_ret].map(
        lambda x: x[: hold_nums] if len(x) > hold_nums else (x + [0] * (hold_nums - len(x))))

    # 计算下周期每天的净值，并扣除手续费得到下周期的实际净值
    temp['下周期每天净值'] = temp[next_ret].apply(lambda x: (np.array(x) + 1).cumprod())
    free_rate = (1 - b_rate) * (1 - s_rate)
    temp['下周期净值'] = temp['下周期每天净值'].apply(lambda x: x[-1] * free_rate)

    # 计算得到每组的资金曲线
    group_nv = temp.groupby(['candle_begin_time', 'groups'])['下周期净值'].mean().reset_index()
    group_nv = group_nv.sort_values(by='candle_begin_time').reset_index(drop=True)
    # 计算每个分组的累计净值
    group_nv['净值'] = group_nv.groupby('groups')['下周期净值'].cumprod()
    group_nv.drop('下周期净值', axis=1, inplace=True)

    # 计算当前数据有多少个分组
    bins = group_nv['groups'].max()

    # 计算各分组在持仓内的每天收益
    group_hold_value = pd.DataFrame(temp.groupby('groups')['下周期每天净值'].mean()).T
    # 所有分组的第一天都是从1开始的
    for col in group_hold_value.columns:
        group_hold_value[col] = group_hold_value[col].apply(lambda x: [1] + list(x))
    # 将未来收益从list展开成逐行的数据
    group_hold_value = group_hold_value.explode(list(group_hold_value.columns)).reset_index(drop=True).reset_index()
    # 重命名列
    group_cols = ['时间'] + [f'第{i}组' for i in range(1, bins + 1)]
    group_hold_value.columns = group_cols
    group_hold_value['offset'] = offset

    print(f'因子分组分析完成，耗时：{datetime.datetime.now() - start_date}')

    # 返回数据：分组资金曲线、分组持仓走势
    return group_nv, group_hold_value


def get_IC_month(IC):
    """
    生成IC月历
    :param IC: IC数据
    :return:
        返回IC月历的df数据
    """

    # resample到月份数据
    IC['candle_begin_time'] = pd.to_datetime(IC['candle_begin_time'])
    IC.set_index('candle_begin_time', inplace=True)
    IC_month = IC.resample('M').agg({'RankIC': 'mean'})
    IC_month.reset_index(inplace=True)
    # 提取出年份和月份
    IC_month['年份'] = IC_month['candle_begin_time'].dt.year.astype('str')
    IC_month['月份'] = IC_month['candle_begin_time'].dt.month
    # 将年份月份设置为index，在将月份unstack为列
    IC_month = IC_month.set_index(['年份', '月份'])['RankIC']
    IC_month = IC_month.unstack('月份')
    IC_month.columns = IC_month.columns.astype(str)
    # 计算各月平均的IC
    IC_month.loc['各月平均', :] = IC_month.mean(axis=0)
    # 按年份大小排名
    IC_month = IC_month.sort_index(ascending=False)

    return IC_month


def get_style_corr(df, factor, offset):
    """
    计算因子的风格暴露
    :param df: df数据，包含因子列和风格列
    :param factor: 因子列
    :param offset: 当前执行的是哪个offset的数据
    :return:
        返回因子的风格暴露的数据
    """

    print('正在进行因子风格暴露分析...')
    start_date = datetime.datetime.now()  # 记录开始时间

    # 取出风格列，格式：以 风格因子_ 开头
    # style_cols = [col for col in df.columns if '风格' in col]
    style_cols = [col for col in df.columns if col.startswith('风格')]
    # 如果df中没有风格因子列，返回空df
    if len(style_cols) == 0:
        return pd.DataFrame()

    # 计算因子与风格的相关系数
    style_corr = df[[factor] + style_cols].corr(method='spearman').iloc[0, 1:].to_frame().reset_index()
    # 整理数据
    style_corr = style_corr.rename(columns={'index': '风格', factor: '相关系数'})
    style_corr['风格'] = style_corr['风格'].map(lambda x: x.split('风格')[1])
    style_corr['offset'] = offset

    print(f'因子风格分析完成，耗时：{datetime.datetime.now() - start_date}')

    return style_corr
