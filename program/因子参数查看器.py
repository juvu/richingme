# -*- coding: utf-8 -*-
"""
中性策略框架 | 邢不行 | 2024分享会
author: 邢不行
微信: xbx6660
"""
from Evaluate import *
from Config import strategy_name
import pandas as pd
pd.set_option('display.max_rows', 1000)
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行

# 动态读取config里面配置的strategy_name脚本
Strategy = __import__('strategy.%s' % strategy_name, fromlist=('',))
hold_period = Strategy.hold_period  # 获取当前策略的持仓周期

# 选择指定offset
offset = 0
# 获取遍历数据
df = pd.read_csv(root_path+f'/data/rtn_{strategy_name}_{hold_period}.csv', encoding='gbk')
# 筛选指定offset
df = df[df['offset'] == offset]

# 画出参数平原图(参数平原支持单因子绘图)
df_1 = df[df['因子数量'] == 1]
if not df_1.empty:
    draw_equity_parameters_plateau(df_1)

# 画出热力图(热力图支持双因子绘图)
df_2 = df[df['因子数量'] == 2]
if not df_2.empty:
    draw_thermodynamic_diagram(df_2)
