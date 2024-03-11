# -*- coding: utf-8 -*-
"""
中性策略框架 | 邢不行 | 2024分享会
author: 邢不行
微信: xbx6660
"""
import pandas as pd
from program.Config import *

# 动态读取config里面配置的strategy_name脚本
Strategy = __import__('program.strategy.%s' % strategy_name, fromlist=('',))

# 从Strategy中获取需要回测持仓周期
hold_period = Strategy.hold_period
# 获取Strategy的文件名
stg_name = Strategy.stg_name
# 判断是否使用现货。如果使用现货，之后保存的文件名中带有SPOT；如果不使用现货，则为SWAP
if Strategy.if_use_spot:
    label = 'SPOT'
else:
    label = 'SWAP'

try:
    evaluate_df = pd.read_csv(back_test_path + f'{stg_name}_{label}_策略评价_{hold_period}.csv', encoding='gbk', index_col=[0])
    evaluate_df.rename(columns={'0': '指标'}, inplace=True)

    equity_df = pd.read_csv(back_test_path + f'{stg_name}_{label}_多空资金曲线_{hold_period}.csv', encoding='gbk')
    equity_df = equity_df[['candle_begin_time', '本周期多空涨跌幅']]
except:
    print('='*50)
    print('请检查一下是否运行了 3_计算多offset资金曲线.py ')
    print('='*50)
    import traceback
    traceback.print_exc()
    exit()

pro_max = equity_df.sort_values('本周期多空涨跌幅', ascending=False).head(5)
pro_min = equity_df.sort_values('本周期多空涨跌幅', ascending=True).head(5)

pro_max['本周期多空涨跌幅'] = pro_max['本周期多空涨跌幅'].apply(lambda x: str(round(100 * x, 2)) + '%')
pro_min['本周期多空涨跌幅'] = pro_min['本周期多空涨跌幅'].apply(lambda x: str(round(100 * x, 2)) + '%')

# 低版本的pandas，不支持 index 参数
# 安装高版本 pandas 命令 ： pip install pandas==1.5.3
tx_evaluate = evaluate_df.to_markdown()

tx_pro_max = pro_max.to_markdown(index=False)
tx_pro_min = pro_min.to_markdown(index=False)

with open(root_path + '/program/发帖脚本/样本模板.txt', 'r', encoding='utf8') as file:
    bbs_post = file.read()
    bbs_post = bbs_post % (tx_evaluate, tx_pro_max, tx_pro_min)
    print(bbs_post)
