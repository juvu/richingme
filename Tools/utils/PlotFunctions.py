# -*- coding: utf-8 -*-
"""
中性策略框架 | 邢不行 | 2024分享会
author: 邢不行
微信: xbx6660
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots
import math


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


# 绘制IC图
def draw_ic_plotly(x, y1, y2, title='', info='', pic_size=[1800, 600]):
    """
    IC画图函数
    :param x: x轴，时间轴
    :param y1: 第一个y轴，每周期的IC
    :param y2: 第二个y轴，累计的IC
    :param title: 图标题
    :param info: IC字符串
    :param pic_size: 图片大小
    :return:
    """

    # 创建子图
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

    # 添加柱状图轨迹
    fig.add_trace(
        go.Bar(
            x=x,  # X轴数据
            y=y1,  # 第一个y轴数据
            name=y1.name,  # 第一个y轴的名字
            marker_color='orange',  # 设置颜色
            marker_line_color='orange'  # 设置柱状图边框的颜色
        ),
        row=1, col=1, secondary_y=False
    )

    # 添加折线图轨迹
    fig.add_trace(
        go.Scatter(
            x=x,  # X轴数据
            y=y2,  # 第二个y轴数据
            text=y2,  # 第二个y轴的文本
            name=y2.name,  # 第二个y轴的名字
            marker_color='blue'  # 设置颜色
        ),
        row=1, col=1, secondary_y=True
    )

    # 更新布局
    fig.update_layout(
        plot_bgcolor='rgb(255, 255, 255)',  # 设置绘图区背景色
        width=pic_size[0],  # 调整宽度
        height=pic_size[1],  # 调整高度
        title={
            'text': title,  # 标题文本
            'x': 0.377,  # 标题相对于绘图区的水平位置
            'y': 0.9,  # 标题相对于绘图区的垂直位置
            'xanchor': 'center',  # 标题的水平对齐方式
            'font': {'color': 'green', 'size': 20}  # 标题的颜色和大小
        },
        xaxis=dict(domain=[0.0, 0.73]),  # 设置 X 轴的显示范围
        legend=dict(
            x=0.8,  # 图例相对于绘图区的水平位置
            y=1.0,  # 图例相对于绘图区的垂直位置
            bgcolor='white',  # 图例背景色
            bordercolor='gray',  # 图例边框颜色
            borderwidth=1  # 图例边框宽度
        ),
        annotations=[
            dict(
                x=x.iloc[len(x) // 2],  # 文字的 x 轴位置
                y=0.6,  # 文字的 y 轴位置
                text=info,  # 文字内容
                showarrow=False,  # 是否显示箭头
                font=dict(
                    size=14  # 设置文字的字体大小
                )
            )
        ],
        hovermode="x unified",
        hoverlabel=dict(bgcolor='rgba(255,255,255,0.5)', )
    )

    # 将图表转换为 HTML 格式
    return_fig = plot(fig, include_plotlyjs=True, output_type='div')

    return return_fig


# 绘制IC月历图
def draw_hot_plotly(x, y, z, title='', pic_size=[1800, 600]):
    """
    IC月历画图函数
    :param x: X轴：月份
    :param y: Y轴：年份
    :param z: Z轴：IC数据
    :param title: IC月历标题名
    :param pic_size: 图片大小
    :return:
        返回IC月历图
    """

    # 创建子图
    fig = make_subplots()

    # 添加热力图轨迹
    fig.add_trace(
        go.Heatmap(
            x=x,  # X轴数据
            y=y,  # Y轴数据
            z=z.values,  # 绘制热力图的数据
            text=z.values,  # 热力图中的数值
            colorscale=[
                [0, 'green'],  # 自定义的颜色点
                [0.5, 'yellow'],
                [1, 'red']
            ],
            colorbar=dict(
                x=0.82,
                y=0.47,
                len=1
            )
        ),
        row=1, col=1
    )

    # 更新布局
    fig.update_layout(
        plot_bgcolor='rgb(255, 255, 255)',  # 设置绘图区背景色
        width=pic_size[0],  # 宽度
        height=pic_size[1],  # 高度
        title={
            'text': title,  # 标题文本
            'x': 0.377,  # 标题相对于绘图区的水平位置
            'y': 0.9,  # 标题相对于绘图区的垂直位置
            'xanchor': 'center',  # 标题的水平对齐方式
            'font': {'color': 'green', 'size': 20}  # 标题的颜色和大小
        },
        xaxis=dict(
            domain=[0.0, 0.73],  # 设置 X 轴的显示范围
            showticklabels=True,
            dtick=1
        )
    )

    z_ = z.applymap(float_num_process, na_action='ignore')

    for i in range(z.shape[1]):
        for j in range(z.shape[0]):
            fig.add_annotation(x=i, y=j, text=z_.iloc[j, i], showarrow=False)

    # 将图表转换为 HTML 格式
    return_fig = plot(fig, include_plotlyjs=True, output_type='div')

    return return_fig


# 绘制柱状图
def draw_bar_plotly(x, y, title='', pic_size=[1800, 600]):
    """
    柱状图画图函数
    :param x: 放到X轴上的数据
    :param y: 放到Y轴上的数据
    :param title: 图标题
    :param pic_size: 图大小
    :return:
        返回柱状图
    """

    # 创建子图
    fig = make_subplots()

    y_ = y.map(float_num_process, na_action='ignore')

    # 添加柱状图轨迹
    fig.add_trace(go.Bar(
        x=x,  # X轴数据
        y=y,  # Y轴数据
        text=y_,  # Y轴文本
        name=x.name  # 图里名字
    ), row=1, col=1)

    # 更新X轴的tick
    fig.update_xaxes(
        tickmode='array',
        tickvals=x
    )

    # 更新布局
    fig.update_layout(
        plot_bgcolor='rgb(255, 255, 255)',  # 设置绘图区背景色
        width=pic_size[0],  # 宽度
        height=pic_size[1],  # 高度
        title={
            'text': title,  # 标题文本
            'x': 0.377,  # 标题相对于绘图区的水平位置
            'y': 0.9,  # 标题相对于绘图区的垂直位置
            'xanchor': 'center',  # 标题的水平对齐方式
            'font': {'color': 'green', 'size': 20}  # 标题的颜色和大小
        },
        xaxis=dict(domain=[0.0, 0.73]),  # 设置 X 轴的显示范围
        showlegend=True,  # 是否显示图例
        legend=dict(
            x=0.8,  # 图例相对于绘图区的水平位置
            y=1.0,  # 图例相对于绘图区的垂直位置
            bgcolor='white',  # 图例背景色
            bordercolor='gray',  # 图例边框颜色
            borderwidth=1  # 图例边框宽度
        )
    )

    # 将图表转换为 HTML 格式
    return_fig = plot(fig, include_plotlyjs=True, output_type='div')

    return return_fig


# 绘制折线图
def draw_line_plotly(x, y1, y2=pd.Series(), update_xticks=False, if_log='False', title='', pic_size=[1800, 600]):
    """
    折线画图函数
    :param x: X轴数据
    :param y1: 左轴数据
    :param y2: 右轴数据
    :param update_xticks: 是否更新x轴刻度
    :param if_log: 是否需要log轴
    :param title: 图标题
    :param pic_size: 图片大小
    :return:
        返回折线图
    """

    # 创建子图
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

    # 添加折线图轨迹
    for col in y1.columns:
        fig.add_trace(
            go.Scatter(
                x=x,  # X轴数据
                y=y1[col],  # Y轴数据
                name=col,  # 图例名字
                line={'width': 2}  # 调整线宽
            ),
            row=1, col=1, secondary_y=False
        )

    if len(y2):
        fig.add_trace(
            go.Scatter(
                x=x,  # X轴数据
                y=y2,  # 第二个Y轴的数据
                name=y2.name,  # 图例名字
                line={'color': 'red', 'dash': 'dot', 'width': 2}  # 调整折现的样式，红色、点图、线宽
            ),
            row=1, col=1, secondary_y=True
        )

    # 如果是画分组持仓走势图的话，更新xticks
    if update_xticks:
        fig.update_xaxes(
            tickmode='array',
            tickvals=x
        )

    # 更新布局
    fig.update_layout(
        plot_bgcolor='rgb(255, 255, 255)',  # 设置绘图区背景色
        width=pic_size[0],
        height=pic_size[1],
        title={
            'text': f'{title}',  # 标题文本
            'x': 0.377,  # 标题相对于绘图区的水平位置
            'y': 0.9,  # 标题相对于绘图区的垂直位置
            'xanchor': 'center',  # 标题的水平对齐方式
            'font': {'color': 'green', 'size': 20}  # 标题的颜色和大小
        },
        xaxis=dict(domain=[0.0, 0.73]),  # 设置 X 轴的显示范围
        legend=dict(
            x=0.8,  # 图例相对于绘图区的水平位置
            y=1.0,  # 图例相对于绘图区的垂直位置
            bgcolor='white',  # 图例背景色
            bordercolor='gray',  # 图例边框颜色
            borderwidth=1  # 图例边框宽度
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor='rgba(255,255,255,0.5)', )
    )
    # 添加log轴
    if if_log:
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
                    ])], )

    # 将图表转换为 HTML 格式
    return_fig = plot(fig, include_plotlyjs=True, output_type='div')

    return return_fig


def draw_double_bar_plotly(x, y1, y2, title='', pic_size=[1800, 600]):
    """
    双柱状图的画图函数
    :param x: X轴数据
    :param y1: 第一个柱状图的数据
    :param y2: 第二个柱状图的数据
    :param title: 标题名
    :param pic_size: 图片大小
    :return:
        返回双柱状图数据
    """

    # 创建子图
    fig = make_subplots()

    # 转换数据，保留小数点位数
    y1_ = y1.map(float_num_process, na_action='ignore')
    y2_ = y2.map(float_num_process, na_action='ignore')

    # 添加第一组柱状图
    fig.add_trace(go.Bar(x=x, y=y1, name=y1.name, text=y1_, marker_color='red'))

    # 添加第二组柱状图
    fig.add_trace(go.Bar(x=x, y=y2, name=y2.name, text=y2_, marker_color='green'))

    # 更新X轴的tick
    fig.update_xaxes(
        tickmode='array',
        tickvals=x
    )

    # 更新布局
    fig.update_layout(
        plot_bgcolor='rgb(255, 255, 255)',  # 设置绘图区背景色
        width=pic_size[0],  # 宽度
        height=pic_size[1],  # 高度
        title={
            'text': title,  # 标题文本
            'x': 0.377,  # 标题相对于绘图区的水平位置
            'y': 0.9,  # 标题相对于绘图区的垂直位置
            'xanchor': 'center',  # 标题的水平对齐方式
            'font': {'color': 'green', 'size': 20}  # 标题的颜色和大小
        },
        xaxis=dict(domain=[0.0, 0.73]),  # 设置 X 轴的显示范围
        legend=dict(
            x=0.8,  # 图例相对于绘图区的水平位置
            y=1.0,  # 图例相对于绘图区的垂直位置
            bgcolor='white',  # 图例背景色
            bordercolor='gray',  # 图例边框颜色
            borderwidth=1  # 图例边框宽度
        )
    )
    # 将图表转换为 HTML 格式
    return_fig = plot(fig, include_plotlyjs=True, output_type='div')

    return return_fig


def merge_html(root_path, fig_list, strategy_file):
    # 创建合并后的网页文件
    merged_html_file = root_path + f'/data/{strategy_file}.html'

    # 创建自定义HTML页面，嵌入fig对象的HTML内容
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8"> 
    <style>
        .body {{
            width: 2000px;
            height:100%;
            }},
        .figure-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
    </style>
    </head>
    <body>
        <h1 style="hight:45px;"></h1>
    <h1 style="margin-left:90px; color: black; font-size: 20px;">{}</h1>
    <h3 style="margin-left:60%; margin-top:10px; font-size: 20px;"><a href="https://bbs.quantclass.cn/thread/31614" target="_blank">如何看懂这些图?</a></h3>
    
    """.format(strategy_file)
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

    res = os.system('start "%s"' % merged_html_file)
    if res != 0:
        os.system('open "%s"' % merged_html_file)
