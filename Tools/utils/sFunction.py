"""
中性策略框架 | 邢不行 | 2024分享会
author: 邢不行
微信: xbx6660
"""
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots
import re
import os
from glob import glob
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def calc_name(filename):
    """
    从文件名中提取出策略名称
    :param filename: 文件名
    :return:
    """
    pattern = r"'(.*?)'"
    matches = re.findall(pattern, filename)
    if len(matches) == 0:
        print('文件名中无法提取策略名称，检查后重试。')
        exit()
    return matches[0]


def format_col(_df):
    """
    统一数据格式
    :param _df:
    :return:
    """
    df = _df.T
    for col_name in ['平均日化收益', '平均累计收益', '胜率_日均', '胜率_累计']:
        if col_name in df.columns:
            df[col_name] = df[col_name].apply(lambda x: '{:.3f}%'.format(x * 100))

    for col_name in ['平均周期数', '平均选中次数']:
        if col_name in df.columns:
            df[col_name] = df[col_name].apply(lambda x: f'{x:.3f}')
    if '选币数' in df.columns:
        df['选币数'] = df['选币数'].apply(lambda x: f'{int(x)}')
    _df = df.T
    return _df


def get_trade_info(symbol_df, direction):
    """
    获取每一笔的交易信息
    :param symbol_df:交易数据
    :param direction: 交易方向
    :return:
    """
    # 先将数据拷贝一份避免污染数据源
    df = symbol_df.copy()

    # 标记周期开始时间
    df.loc[df['open_signal'].notnull(), 'open_time'] = df['candle_begin_time']
    df['open_time'].fillna(method='ffill', inplace=True)

    # 针对周期开始时间进行分组
    groups = df.groupby('open_time')

    res_list = []  # 存放每个周期的分组结果
    # 遍历每一个分组
    for t, g in groups:
        # 获取这笔交易的结束的索引
        end_inx = g[g['close_signal'] == 0].index.min()
        if pd.isnull(end_inx):
            # 能进到这里的老板，说明自己改过框架
            end_inx = g.index.max()
        # 截取数据
        g = g[g.index <= end_inx]

        # 计算这笔交易的信息
        res = pd.DataFrame()
        res.loc[0, 'open_time'] = t  # 交易开始时间
        res.loc[0, 'close_time'] = g['candle_begin_time'].iloc[-1]  # 交易结束时间
        res.loc[0, 'open_price'] = g['open'].iloc[0]  # 开仓价格
        res.loc[0, 'close_price'] = g['close'].iloc[-1]  # 平仓价格

        res_list.append(res)

    # 聚合每一笔交易
    trade_df = pd.concat(res_list, ignore_index=True)
    # 添加交易方向
    trade_df['direction'] = 'long' if direction == '多头' else 'short'
    # 计算交易收益
    if direction == '多头':
        trade_df['return'] = trade_df['close_price'] / trade_df['open_price'] - 1
    else:
        trade_df['return'] = 1 - trade_df['close_price'] / trade_df['open_price']

    # 将收益率转为为百分比格式
    trade_df['return'] = trade_df['return'].apply(lambda x: str(round(100 * x, 2)) + '%')
    return trade_df


def draw_hedge_signal_plotly(df, save_path, title, res_loc, day_df, trade_df, add_factor_main_list,
                             add_factor_sub_list, color_dict, pic_size=[1880, 1656]):
    """
    绘制k线图以及指标图

    :param df:              原始数据
    :param save_path:       保存图片的路径
    :param title:           图片标题
    :param day_df:           日线数据
    :param pic_size:        图片大小
    """

    # 随机颜色的列表
    color_list = ['#feb71d', '#dc62af', '#4d50bb', '#f0eb8d', '#018b96', '#e7adea']
    color_i = 0
    for each_factor in add_factor_main_list:
        if each_factor['因子名称'] not in color_dict.keys():
            color_dict[each_factor['因子名称']] = color_list[color_i % len(color_list)]
            color_i += 1
    for each_sub in add_factor_sub_list:
        for each_factor in each_sub['因子名称']:
            if each_factor not in color_dict.keys():
                color_dict[each_factor] = color_list[color_i % len(color_list)]
                color_i += 1

    time_data = df['candle_begin_time']
    add_rows = len(add_factor_sub_list)

    # 700是主图，附图200。
    pic_size[1] = 700 * 2 + 200 * add_rows

    # 主图有没有副轴
    have_secondary_y = any(each_factor.get('次坐标轴', False) for each_factor in add_factor_main_list)

    # 构建画布，因为最后一个图会有不shareX的问题，所以这里先不设置shared_xaxes
    fig = make_subplots(rows=2 + add_rows, cols=1,
                        specs=[[{"secondary_y": have_secondary_y}]] + (add_rows + 1) * [[{"secondary_y": False}]])

    # 绘制k线图
    # =====主图主要数据
    fig.add_trace(go.Candlestick(
        x=time_data,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='k线',
    ), row=1, col=1)

    # 绘制主图上其它因子（包括指数或者均线）
    for each_factor in add_factor_main_list:
        if each_factor['因子名称'] == 'btc':
            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=df['btc_close'],
                    name='btc',
                    marker_color=color_dict['btc']
                ),
                row=1, col=1, secondary_y=each_factor['次坐标轴']
            )

        else:
            if 'period' in each_factor:
                f_name = each_factor['因子名称'] + '_' + each_factor['period']
            else:
                f_name = each_factor['因子名称']

            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=df[f_name],
                    name=each_factor['因子名称'],
                    marker_color=color_dict[each_factor['因子名称']]
                ),
                row=1, col=1, secondary_y=each_factor['次坐标轴']
            )

    # 绘制主图买卖点
    mark_point_list = []
    for i in df[(df['open_signal'].notna()) | (df['close_signal'].notna())].index:
        # 获取买卖点信息
        open_signal = df.loc[i, 'open_signal']
        close_signal = df.loc[i, 'close_signal']
        # 只有开仓信号，没有平仓信号
        if pd.notnull(open_signal) and pd.isnull(close_signal):
            signal = str(int(open_signal))
        # 没有开仓信号，只有平仓信号
        elif pd.isnull(open_signal) and pd.notnull(close_signal):
            signal = str(int(close_signal))
        else:  # 同时有开仓信号和平仓信号
            signal = str(int(open_signal)) + '_' + str(int(close_signal))
        # 标记买卖点，在最高价上方标记
        y = df.at[i, 'high']
        mark_point_list.append({
            'x': df.at[i, 'candle_begin_time'],
            'y': y,
            'showarrow': True,
            'text': signal,
            'arrowside': 'end',
            'arrowhead': 6,
        })
    # 更新画布布局，把买卖点标记上、把主图的大小调整好
    fig.update_layout(annotations=mark_point_list, template="none", width=pic_size[0], height=pic_size[1],
                      title_text=title, hovermode='x',
                      yaxis=dict(domain=[1 - 700 / pic_size[1], 1.0]), xaxis=dict(domain=[0.0, 0.73]),
                      xaxis_rangeslider_visible=False,
                      )
    # 主图有副轴，就更新
    if have_secondary_y:
        fig.update_layout(yaxis2=dict(domain=[1 - 700 / pic_size[1], 1.0]), xaxis2=dict(domain=[0.0, 0.73]))

    # =====附图数据
    # ==绘制子图
    row = 2  # 1是第一个主图，所以不用管
    # 子图的范围都做算好
    y_domains = [[1 - (900 + 200 * i) / pic_size[1], 1 - (700 + 200 * i) / pic_size[1]] for i in
                 range(0, add_rows)]
    x_domains = [[0.0, 0.73] for _ in range(0, add_rows)]

    # 做每个子图
    for each_factor in add_factor_sub_list:
        graphicStyle = each_factor['图形样式'].upper()
        for each_sub_factor in each_factor['因子名称']:
            if 'period' in each_factor:
                f_name = each_sub_factor + '_' + each_factor['period']
            else:
                f_name = each_sub_factor
            if graphicStyle == '柱状图':
                fig.add_trace(go.Bar(x=time_data, y=df[f_name], name=f_name,
                                     marker_color=color_dict[each_sub_factor]), row=row, col=1)
            elif graphicStyle == '折线图':
                if ('period' in each_factor) and (each_factor['period'] != 'H'):
                    # 日线的因子，fillna变成H，否则折线图显示不出
                    df[f_name + '_D2H'] = df[f_name].copy()
                    df[f_name + '_D2H'].fillna(method='ffill', inplace=True)
                    fig.add_trace(go.Scatter(x=time_data, y=df[f_name + '_D2H'], name=f_name,
                                             marker_color=color_dict[each_sub_factor]), row=row, col=1)
                else:
                    fig.add_trace(go.Scatter(x=time_data, y=df[f_name], name=f_name,
                                             marker_color=color_dict[each_sub_factor]), row=row, col=1)

            elif graphicStyle == 'K线图':
                fig.add_trace(
                    go.Candlestick(
                        x=time_data,
                        open=df['btc_open'],  # 字段数据必须是元组、列表、numpy数组、或者pandas的Series数据
                        high=df['btc_high'],
                        low=df['btc_low'],
                        close=df['btc_close'],
                        name='btc'
                    ),
                    row=row, col=1
                )

                fig.update_xaxes(rangeslider_visible=False, row=row, col=1)
        fig.update_yaxes(dict(domain=y_domains[row - 2]), row=row)
        fig.update_xaxes(dict(domain=x_domains[row - 2]), row=row)
        fig.update_yaxes(title_text='、'.join(
            each_factor['因子名称']) + f' {each_factor["period"] if "period" in each_factor else ""}', row=row, col=1)
        row += 1

    # ==绘制日线
    time_data = day_df['candle_begin_time']
    # 绘制k线图
    fig.add_trace(go.Candlestick(
        x=time_data,
        open=day_df['open'],  # 字段数据必须是元组、列表、numpy数组、或者pandas的Series数据
        high=day_df['high'],
        low=day_df['low'],
        close=day_df['close'],
        name='日频k线',
    ), row=2 + add_rows, col=1)
    fig.update_xaxes(rangeslider_visible=False, row=2 + add_rows, col=1)
    # mark_point_list3 = []
    for i in day_df[day_df['signal'].notna()].index:
        # 标记开始时间
        y = day_df.at[i, 'high']
        # 在周期数据上标记
        mark_point_list.append({
            'x': (day_df.at[i, 'candle_begin_time'] + pd.to_timedelta('23H') if day_df.loc[i, 'signal'] == 'end' else
                  day_df.at[i, 'candle_begin_time']),
            'y': y,
            'showarrow': True,
            'text': day_df.loc[i, 'signal'],
            'arrowside': 'end',
            'arrowhead': 6,
        })
        # 在日线数据上标记
        mark_point_list.append({
            'x': day_df.at[i, 'candle_begin_time'],
            'y': y,
            'showarrow': True,
            'text': day_df.loc[i, 'signal'],
            'arrowside': 'end',
            'arrowhead': 6,
            'xref': f'x{add_rows + 2}',
            'yref': f'y{add_rows + 3 if have_secondary_y else 2}'
        })
    # 更新画布布局、把起止时间点标上
    fig.update_yaxes(dict(domain=[0, 1 - (700 + 200 * add_rows) / pic_size[1]]), row=2 + add_rows)
    fig.update_xaxes(dict(domain=[0.0, 0.73]), row=2 + add_rows)
    fig.update_layout(annotations=mark_point_list, title=title)
    # ===收益图
    res_loc['每周期平均收益（日化）'] = str(round(100 * res_loc['每周期平均收益（日化）'], 3)) + '%'
    res_loc['区间累计收益'] = str(round(100 * res_loc['区间累计收益'], 3)) + '%'
    res_loc['首次选中时间'] = res_loc['首次选中时间'].strftime('%Y-%m-%d %H:%M:%S')
    res_loc['最后选中时间'] = res_loc['最后选中时间'].strftime('%Y-%m-%d %H:%M:%S')
    del res_loc['选中周期']

    res_trace = go.Table(header=dict(values=list(['项目', '情况'])),
                         cells=dict(
                             values=[res_loc.index.tolist(), res_loc.values.tolist()]),
                         domain=dict(x=[0.85, 1.0], y=[0.85, 1.0]))
    fig.add_trace(res_trace)

    # ===持仓时间图
    table_trace = go.Table(header=dict(values=list(['open_time', 'close_time', 'return'])),
                           cells=dict(
                               values=[trade_df['open_time'], trade_df['close_time'], trade_df['return']]),
                           domain=dict(x=[0.75, 1.0], y=[0.3, 0.8]))
    fig.add_trace(table_trace)

    # 除了最后的日线图，其余统统同步X轴
    for i in range(1, 2 + add_rows):
        fig.update_xaxes(matches='x', row=i, col=1)

    # 图例调整位置
    fig.update_layout(legend=dict(x=0.75, y=1))
    # 保存路径
    save_path = save_path + title + '.html'
    plot(figure_or_data=fig, filename=save_path, auto_open=False)


def read_coin_factor(root_path, hold_period, all_factor_list, symbol, symbol_type):
    factor_period = hold_period[-1]
    parent_directory = os.path.join(root_path, 'data', '数据整理', symbol_type, symbol, 'factors')
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
    # df = read_factor(factor_path, symbol, factor_dict, factor_period)
    df = pd.DataFrame()

    # 获取这个目录下的因子信息
    for factor, factor_col in factor_dict.items():
        factor_path = os.path.join(parent_directory, f'{symbol}_{factor}_{factor_period}.pkl')
        if os.path.exists(factor_path):
            factor_df = pd.read_feather(factor_path)
            factor_col = [x for x in factor_col if x in factor_df.columns]
            if df.empty:
                df = factor_df[['candle_begin_time'] + factor_col].copy()
            else:
                # merge
                df = pd.merge(df, factor_df[['candle_begin_time'] + factor_col], 'left', 'candle_begin_time')
    if df.empty:
        return df
    all_factor_list = [x for x in all_factor_list if x in df.columns]
    df.rename(columns={key: f'{key}_{factor_period}' for key in all_factor_list}, inplace=True)
    df = df[['candle_begin_time'] + [f'{x}_{factor_period}' for x in all_factor_list]]
    return df
