# -*- coding: utf-8 -*-
"""
中性策略框架 | 邢不行 | 2024分享会
author: 邢不行
微信: xbx6660
"""
import os
import re
import itertools
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from Config import root_path
from matplotlib import pyplot as plt
from plotly.offline import plot
from plotly.subplots import make_subplots
import plotly.express as px


def merge_html(root_path, fig_list, strategy_file):
    # 创建合并后的网页文件
    merged_html_file = root_path + f'/data/{strategy_file}汇总.html'

    # 创建自定义HTML页面，嵌入fig对象的HTML内容
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8">
    <style>
        .figure-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
    </style>
    </head>
    <body>"""
    for fig in fig_list:
        html_content += f"""
        <div class="figure-container">
            {fig}
        </div>
        """
    html_content += '</body> </html>'

    # 保存自定义HTML页面
    with open(merged_html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    res = os.system('start %s' % merged_html_file)
    if res != 0:
        os.system('open %s' % merged_html_file)


def draw_bar_plotly(long_result, short_result, x, y, title, text, pic_size=[1200, 500]):
    """
    绘制柱状图
    :param long_result: 多头数据
    :param short_result: 空头数据
    :param x: x轴名称
    :param y: y轴名称
    :param title: 图片标题
    :param text: 图片说明
    :param path: 保存图片路径
    :param show: 是否显示柱状图
    :return:
    """
    # 配置fig的列表，生成图之后先添加到fig_list中，最后一起画图
    fig_list = []

    # 多头的分箱图
    bar_fig = px.bar(long_result.loc[long_result.index == long_result.index[-2]].round(2), x=x, y=y, title='多头' + title, text=text, width=pic_size[0], height=pic_size[1])
    return_fig = plot(bar_fig, include_plotlyjs=True, output_type='div')
    fig_list.append(return_fig)

    # 空头的分箱图
    bar_fig = px.bar(short_result.loc[short_result.index == short_result.index[-2]].round(2), x=x, y=y, title='空头' + title, text=text, width=pic_size[0], height=pic_size[1])
    return_fig = plot(bar_fig, include_plotlyjs=True, output_type='div')
    fig_list.append(return_fig)

    # 将多头分箱图和空头分箱图一起绘制
    merge_html(root_path, fig_list, strategy_file='分箱图')


def robustness_test(long_df, short_df, bins=10, long_factor='因子', short_factor='因子'):
    # 如果bins=0，则不进行分箱图
    if bins == 0:
        return

    def get_group_result(df, factor_name='因子'):
        # 对币种排序并分组
        df['rank'] = df.groupby('candle_begin_time')[factor_name].rank(method='first')
        df['总币数'] = df.groupby('candle_begin_time')['symbol'].transform('size')
        df = df[df['总币数'] >= bins]
        df['group'] = df.groupby('candle_begin_time')['rank'].transform(
            lambda x: pd.qcut(x, q=bins, labels=range(1, bins + 1), duplicates='drop'))
        # 计算净值数据
        df['收益'] = df['ret_next'] + 1
        result = df.groupby(['candle_begin_time', 'group'])['收益'].mean().to_frame()
        result.reset_index('group', inplace=True)
        result['group'] = result['group'].astype(str)
        result['asset'] = result.groupby('group')['收益'].cumprod()

        return result

    # 对头空头分别获取分箱图的结果
    long_result = get_group_result(long_df, factor_name=long_factor)
    short_result = get_group_result(short_df, factor_name=short_factor)

    # 对多头结果和空头结果 分别绘制策略分箱柱状图
    draw_bar_plotly(long_result, short_result, x='group', y='asset', title=f'{bins}分箱 资金曲线', text='asset')
    print('分箱测试完毕！')


# 计算策略评价指标
def strategy_evaluate(equity, net_col='多空资金曲线', pct_col='本周期多空涨跌幅', turnover_col='多空调仓比例'):
    """
    回测评价函数
    :param equity: 资金曲线数据
    :param net_col: 资金曲线列名
    :param pct_col: 周期涨跌幅列名
    :param turnover_col: 调仓比例列名
    :return:
    """
    # ===新建一个dataframe保存回测指标
    results = pd.DataFrame()

    # 将数字转为百分数
    def num_to_pct(value):
        return '%.2f%%' % (value * 100)

    # ===计算累积净值
    results.loc[0, '累积净值'] = round(equity[net_col].iloc[-1], 2)

    # ===计算年化收益
    annual_return = (equity[net_col].iloc[-1]) ** (
            '1 days 00:00:00' / (equity['candle_begin_time'].iloc[-1] - equity['candle_begin_time'].iloc[0]) * 365) - 1
    results.loc[0, '年化收益'] = num_to_pct(annual_return)

    # ===计算最大回撤，最大回撤的含义：《如何通过3行代码计算最大回撤》https://mp.weixin.qq.com/s/Dwt4lkKR_PEnWRprLlvPVw
    # 计算当日之前的资金曲线的最高点
    equity[f'{net_col.split("资金曲线")[0]}max2here'] = equity[net_col].expanding().max()
    # 计算到历史最高值到当日的跌幅，drowdwon
    equity[f'{net_col.split("资金曲线")[0]}dd2here'] = equity[net_col] / equity[f'{net_col.split("资金曲线")[0]}max2here'] - 1
    # 计算最大回撤，以及最大回撤结束时间
    end_date, max_draw_down = tuple(equity.sort_values(by=[f'{net_col.split("资金曲线")[0]}dd2here']).iloc[0][['candle_begin_time', f'{net_col.split("资金曲线")[0]}dd2here']])
    # 计算最大回撤开始时间
    start_date = equity[equity['candle_begin_time'] <= end_date].sort_values(by=net_col, ascending=False).iloc[0]['candle_begin_time']
    results.loc[0, '最大回撤'] = num_to_pct(max_draw_down)
    results.loc[0, '最大回撤开始时间'] = str(start_date)
    results.loc[0, '最大回撤结束时间'] = str(end_date)
    # ===年化收益/回撤比：我个人比较关注的一个指标
    results.loc[0, '年化收益/回撤比'] = round(annual_return / abs(max_draw_down), 2)
    # ===统计每个周期
    results.loc[0, '盈利周期数'] = len(equity.loc[equity[pct_col] > 0])  # 盈利笔数
    results.loc[0, '亏损周期数'] = len(equity.loc[equity[pct_col] <= 0])  # 亏损笔数
    results.loc[0, '胜率'] = num_to_pct(results.loc[0, '盈利周期数'] / len(equity))  # 胜率
    results.loc[0, '每周期平均收益'] = num_to_pct(equity[pct_col].mean())  # 每笔交易平均盈亏
    results.loc[0, '盈亏收益比'] = round(equity.loc[equity[pct_col] > 0][pct_col].mean() / equity.loc[equity[pct_col] <= 0][pct_col].mean() * (-1), 2)  # 盈亏比
    if 1 in equity['是否爆仓'].to_list():
        results.loc[0, '盈亏收益比'] = 0
    results.loc[0, '单周期最大盈利'] = num_to_pct(equity[pct_col].max())  # 单笔最大盈利
    results.loc[0, '单周期大亏损'] = num_to_pct(equity[pct_col].min())  # 单笔最大亏损

    # ===连续盈利亏损
    results.loc[0, '最大连续盈利周期数'] = max(
        [len(list(v)) for k, v in itertools.groupby(np.where(equity[pct_col] > 0, 1, np.nan))])  # 最大连续盈利次数
    results.loc[0, '最大连续亏损周期数'] = max(
        [len(list(v)) for k, v in itertools.groupby(np.where(equity[pct_col] <= 0, 1, np.nan))])  # 最大连续亏损次数

    # ===其他评价指标
    results.loc[0, '每周期平均换手率'] = num_to_pct(equity[turnover_col].mean())
    results.loc[0, '收益率标准差'] = num_to_pct(equity[pct_col].std())

    # ===每年、每月收益率
    temp = equity.copy()
    temp.set_index('candle_begin_time', inplace=True)
    year_return = temp[[pct_col]].resample(rule='A').apply(lambda x: (1 + x).prod() - 1)
    month_return = temp[[pct_col]].resample(rule='M').apply(lambda x: (1 + x).prod() - 1)

    def num2pct(x):
        if str(x) != 'nan':
            return str(round(x * 100, 2)) + '%'
        else:
            return x

    year_return['涨跌幅'] = year_return[pct_col].apply(num2pct)

    # 对每月收益进行处理，做成二维表
    month_return.reset_index(inplace=True)
    month_return['year'] = month_return['candle_begin_time'].dt.year
    month_return['month'] = month_return['candle_begin_time'].dt.month
    month_return.set_index(['year', 'month'], inplace=True)
    del month_return['candle_begin_time']
    month_return_all = month_return[pct_col].unstack()
    month_return_all.loc['mean'] = month_return_all.mean(axis=0)
    month_return_all = month_return_all.apply(lambda x: x.apply(num2pct))

    return results.T, year_return[['涨跌幅']], month_return_all


def draw_equity_curve_plotly(df, data_dict, date_col=None, right_axis=None, pic_size=[1500, 800], chg=False,
                             title=None, path=root_path + '/data/pic.html', show=True):
    """
    绘制策略曲线
    :param df: 包含净值数据的df
    :param data_dict: 要展示的数据字典格式：｛图片上显示的名字:df中的列名｝
    :param date_col: 时间列的名字，如果为None将用索引作为时间列
    :param right_axis: 右轴数据 ｛图片上显示的名字:df中的列名｝
    :param pic_size: 图片的尺寸
    :param chg: datadict中的数据是否为涨跌幅，True表示涨跌幅，False表示净值
    :param title: 标题
    :param path: 图片路径
    :param show: 是否打开图片
    :return:
    """
    draw_df = df.copy()

    # 设置时间序列
    if date_col:
        time_data = draw_df[date_col]
    else:
        time_data = draw_df.index

    # 绘制左轴数据
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for key in data_dict:
        if chg:
            draw_df[data_dict[key]] = (draw_df[data_dict[key]] + 1).fillna(1).cumprod()
        fig.add_trace(go.Scatter(x=time_data, y=draw_df[data_dict[key]], name=key, ))

    # 绘制右轴数据
    if right_axis:
        key = list(right_axis.keys())[0]
        fig.add_trace(go.Scatter(x=time_data, y=draw_df[right_axis[key]], name=key + '(右轴)',
                                 #  marker=dict(color='rgba(220, 220, 220, 0.8)'),
                                 marker_color='orange',
                                 opacity=0.1, line=dict(width=0),
                                 fill='tozeroy',
                                 yaxis='y2'))  # 标明设置一个不同于trace1的一个坐标轴
        for key in list(right_axis.keys())[1:]:
            fig.add_trace(go.Scatter(x=time_data, y=draw_df[right_axis[key]], name=key + '(右轴)',
                                     #  marker=dict(color='rgba(220, 220, 220, 0.8)'),
                                     opacity=0.1, line=dict(width=0),
                                     fill='tozeroy',
                                     yaxis='y2'))  # 标明设置一个不同于trace1的一个坐标轴

    fig.update_layout(template="none", width=pic_size[0], height=pic_size[1], title_text=title,
                      hovermode="x unified", hoverlabel=dict(bgcolor='rgba(255,255,255,0.5)', ),
                      )
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(label="线性 y轴",
                         method="relayout",
                         args=[{"yaxis.type": "linear"}]),
                    dict(label="Log y轴",
                         method="relayout",
                         args=[{"yaxis.type": "log"}]),
                ])],
    )
    plot(figure_or_data=fig, filename=path, auto_open=False)

    fig.update_yaxes(
        showspikes=True, spikemode='across', spikesnap='cursor', spikedash='solid', spikethickness=1,  # 峰线
    )
    fig.update_xaxes(
        showspikes=True, spikemode='across+marker', spikesnap='cursor', spikedash='solid', spikethickness=1,  # 峰线
    )

    # 打开图片的html文件，需要判断系统的类型
    if show:
        res = os.system('start %s' % path)
        if res != 0:
            os.system('open %s' % path)


def draw_thermodynamic_diagram(df, factor_count=2, show=True):
    """
    绘制热力图
    :param df: 原数据
    :param factor_count: 因子数量
    :param show: 是否显示热力图
    :return:
    """
    def analysis(x):
        x = eval(x)
        series_list = []
        for k, v in x:
            series_list.append(k)
            series_list.append(v)
        return pd.Series(series_list)

    para = df['因子组合'].apply(analysis)
    columns_list = []
    for _ in range(factor_count):
        columns_list += ['因子%s' % _, '因子%s_reverse' % _]
    df[columns_list] = para
    conde1 = f'因子0_reverse==True & 因子1_reverse==True'
    conde2 = f'因子0_reverse==False & 因子1_reverse==True'
    conde3 = f'因子0_reverse==False & 因子1_reverse==False'
    conde4 = f'因子0_reverse==True & 因子1_reverse==False'
    cond_list = [conde1, conde2, conde3, conde4]

    for _ in range(len(cond_list)):
        hot_df = df.query(cond_list[_])[['因子0', '因子1', '年化收益/回撤比']].copy()
        hot_df['因子0数值'] = hot_df['因子0'].apply(lambda x: int(x.split('_')[1]))
        hot_df['因子1数值'] = hot_df['因子1'].apply(lambda x: int(x.split('_')[1]))
        hot_df.sort_values(by=['因子0数值', '因子1数值'], inplace=True)
        hot_df.reset_index(inplace=True, drop=True)
        layout = go.Layout(
            # plot_bgcolor='red',  # 图背景颜色
            paper_bgcolor='white',  # 图像背景颜色
            autosize=True,
            # width=2000,
            # height=1200,
            title=cond_list[_],
            titlefont=dict(size=30, color='gray'),

            # 图例相对于左下角的位置
            legend=dict(
                x=0.02,
                y=0.02
            ),

            # x轴的刻度和标签
            xaxis=dict(title='因子0',  # 设置坐标轴的标签
                       titlefont=dict(color='red', size=20),
                       tickfont=dict(color='blue', size=18, ),
                       tickangle=45,  # 刻度旋转的角度
                       showticklabels=True,  # 是否显示坐标轴
                       # 刻度的范围及刻度
                       # autorange=False,
                       # range=[0, 100],
                       # type='linear',
                       ),

            # y轴的刻度和标签
            yaxis=dict(title='因子1',  # 坐标轴的标签
                       titlefont=dict(color='blue', size=18),  # 坐标轴标签的字体及颜色
                       tickfont=dict(color='green', size=20, ),  # 刻度的字体大小及颜色
                       showticklabels=True,  # 设置是否显示刻度
                       tickangle=-45,
                       # 设置刻度的范围及刻度
                       autorange=True,
                       # range=[0, 100],
                       # type='linear',
                       ),
        )

        fig = go.Figure(data=go.Heatmap(
            showlegend=True,
            name='数据',
            x=hot_df['因子0'],
            y=hot_df['因子1'],
            z=hot_df['年化收益/回撤比'],
            type='heatmap',
        ),
            layout=layout
        )

        fig.update_layout(margin=dict(t=100, r=150, b=100, l=100), autosize=True)

        path = f'{root_path}/data/{_}.html'
        plot(figure_or_data=fig, filename=path, auto_open=False)

        # 打开图片的html文件，需要判断系统的类型
        if show:
            res = os.system('start %s' % path)
            if res != 0:
                os.system('open %s' % path)


def draw_equity_parameters_plateau(df, show=True):
    """
    绘制参数平原
    :param df: 原数据
    :param show: 是否显示热力图
    :return:
    """
    def analysis(x):
        x = eval(x)
        series_list = []
        for k, v in x:
            series_list.append(k)
            series_list.append(v)
        return pd.Series(series_list)

    para = df['因子组合'].apply(analysis)
    df[['因子1', '排序']] = para
    df[['因子', '因子参数']] = df['因子1'].str.split('_', expand=True)
    for _key, _group in df.groupby(['因子', '排序']):
        pic_title = f'因子_{_key[0]}_{_key[1]}'
        _temp = _group.copy()
        _temp['因子参数'] = _temp['因子参数'].astype(int)
        _temp.sort_values('因子参数', inplace=True)
        _temp.reset_index(inplace=True, drop=True)

        bar_fig = px.bar(_temp, x='因子参数', y='累积净值', title=pic_title, text='累积净值')
        plot(figure_or_data=bar_fig, filename=root_path + f'/data/{pic_title}.html', auto_open=False)
        # 打开图片的html文件，需要判断系统的类型
        res = os.system('start %s' % (root_path + f'/data/{pic_title}.html'))
        if res != 0:
            os.system('open %s' % (root_path + f'/data/{pic_title}.html'))