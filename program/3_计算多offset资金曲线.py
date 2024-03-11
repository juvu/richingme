# -*- coding: utf-8 -*-
"""
中性策略框架 | 邢不行 | 2024分享会
author: 邢不行
微信: xbx6660
"""
import pandas as pd
from Config import *
from Evaluate import *
import sys

# 动态读取config里面配置的strategy_name脚本
Strategy = __import__('strategy.%s' % strategy_name, fromlist=('',))

# 获取当前环境下的python解释器
python_exec = sys.executable
# 指定脚本名称
exec_file_name = '2_选币_单offset.py'
# 从Strategy中获取需要回测的factor_list
factor_list = Strategy.factor_list
# 从Strategy中获取需要回测持仓周期
hold_period = Strategy.hold_period
# 获取Strategy的文件名
stg_name = Strategy.stg_name
# 获取Strategy的if_use_spot
if_use_spot = Strategy.if_use_spot
# 获取当前持仓周期下的所有offset
offset_list = list(range(0, int(hold_period[:-1])))

# 判断是否使用现货。如果使用现货，之后保存的文件名中带有SPOT；如果不使用现货，则为SWAP
if if_use_spot:
    label = 'SPOT'
else:
    label = 'SWAP'


# =====遍历运行单offset资金曲线
def exec_command(_offset):
    # 使用os进行调用python运行指定脚本
    command = '%s %s %s %s' % (python_exec, exec_file_name, hold_period, _offset)
    print(command)
    # 运行命令
    os.system(command)


# 并行或串行，遍历所有offset进行遍历
multiply_process = False  # 是否并行。有点消耗内存哦，根据电脑配置决定是否使用并行。并行计算，输出结果会混乱，主要看哪个核心快，与程序无关
if multiply_process:
    Parallel(n_jobs=n_jobs)(
        delayed(exec_command)(_offset)
        for _offset in offset_list
    )
else:
    for _offset in offset_list:
        exec_command(_offset)


# =====整合所有offset资金曲线
filepath_list = os.listdir(back_test_path)  # 获取回测结果目录下的文件
if if_use_spot:
    label = 'SPOT'
else:
    label = 'SWAP'
filepath_list = [_ for _ in filepath_list if _.startswith(f'{stg_name}_{label}_资金曲线_{hold_period}')]  # 获取指定因子的文件信息

all_equity_df = pd.DataFrame()  # 定义空的资金曲线df
rtn_list = []  # 所有offset策略评价集合
sorted_offset = {}  # 用于图表的offset排序
# 遍历所有offset资金曲线
for _filepath in filepath_list:
    _offset = _filepath.split('_')[-1].replace('.csv', '')  # 从文件名称中解析出offset
    # 读取资金曲线文件
    _df = pd.read_csv(back_test_path + _filepath, encoding='gbk', parse_dates=['candle_begin_time'])
    _df = _df[['candle_begin_time', '涨跌幅', '多空资金曲线', '多空调仓比例', '是否爆仓']]  # 获取指定字段
    _df.rename(columns={'涨跌幅': f'涨跌幅_{_offset}', '多空资金曲线': f'多空资金曲线_{_offset}',
                        '多空调仓比例': f'多空调仓比例_{_offset}', '是否爆仓': f'是否爆仓_{_offset}'}, inplace=True)

    # 将各个offset资金曲线合并到all_equity_df中
    if all_equity_df.empty:  # 数据为空，直接拷贝读取的资金曲线文件即可
        all_equity_df = _df.copy()
    else:  # 当前已经存在资金曲线，则将新offset资金曲线合并到all_equity_df中
        all_equity_df = pd.merge(left=all_equity_df, right=_df, on='candle_begin_time', how='outer')

    # 策略评价
    _df['本周期多空涨跌幅'] = _df[f'涨跌幅_{_offset}']
    _df['多空资金曲线'] = _df[f'多空资金曲线_{_offset}']
    _df['涨跌幅'] = _df[f'涨跌幅_{_offset}']
    _df['多空调仓比例'] = _df[f'多空调仓比例_{_offset}']
    _df['是否爆仓'] = _df[f'是否爆仓_{_offset}']
    rtn, _, __ = strategy_evaluate(_df)
    rtn = rtn.T
    rtn['offset'] = _offset
    rtn_list.append(rtn)
    sorted_offset[f'多空资金曲线_{_offset}'] = _df.iloc[-1]['多空资金曲线']

# 合并每个offset的策略评价信息
rtn_df = pd.concat(rtn_list, ignore_index=True)
rtn_df.set_index('offset', inplace=True)
rtn_df.sort_values('累积净值', ascending=False, inplace=True)
print(rtn_df.to_markdown())  # 输出各个offset策略评价信息

# 重新排序
all_equity_df.sort_values('candle_begin_time', inplace=True)
all_equity_df.reset_index(inplace=True, drop=True)

# 对空的涨跌幅填充0
pct_cols = [_ for _ in all_equity_df.columns if '涨跌幅' in _]
all_equity_df.loc[:, pct_cols] = all_equity_df[pct_cols].fillna(value=0)

# 对空的资金曲线填充1
equity_cols = [_ for _ in all_equity_df.columns if '多空资金曲线' in _]
all_equity_df.loc[:, equity_cols] = all_equity_df[equity_cols].fillna(value=1)

# 获取多空调仓比例的列名
turnover_cols = [_ for _ in all_equity_df.columns if '多空调仓比例' in _]

# 获取是否爆仓的列名
warehouse_cols = [_ for _ in all_equity_df.columns if '是否爆仓' in _]

# 计算所有offset的均值，获取账户总的净值资金曲线
all_equity_df['多空资金曲线'] = all_equity_df[equity_cols].mean(axis=1)
# 计算账户涨跌幅，通过账户净值反推
all_equity_df['本周期多空涨跌幅'] = pd.DataFrame([1] + all_equity_df['多空资金曲线'].to_list()).pct_change()[0].iloc[1:].to_list()
all_equity_df['涨跌幅'] = all_equity_df['本周期多空涨跌幅']
# 计算周期平均多空调仓比例
all_equity_df['多空调仓比例'] = all_equity_df[turnover_cols].mean(axis=1)
all_equity_df['是否爆仓'] = all_equity_df[warehouse_cols].sum(axis=1)
save_path = os.path.join(back_test_path, f'{stg_name}_{label}_多空资金曲线_{hold_period}.csv')
all_equity_df.to_csv(save_path, encoding='gbk', index=False)

# 对所有offset资金曲线进行评价
rtn, year_return, month_return = strategy_evaluate(all_equity_df)
save_path = os.path.join(back_test_path, f'{stg_name}_{label}_策略评价_{hold_period}.csv')
rtn.to_csv(save_path, encoding='gbk')
print('\n\n所有offset综合策略评价\n', rtn, '\n\n所有offset综合分年收益率：\n', year_return, '\n\n所有offset综合分月收益率：\n', month_return)

# =====画图
BTC = pd.read_csv(swap_path + 'BTC-USDT.csv', encoding='gbk', parse_dates=['candle_begin_time'], skiprows=1)
BTC['BTC涨跌幅'] = BTC['close'].pct_change()
all_equity_df = pd.merge(left=all_equity_df, right=BTC[['candle_begin_time', 'BTC涨跌幅']], on=['candle_begin_time'], how='left')
all_equity_df['BTC涨跌幅'].fillna(value=0, inplace=True)
all_equity_df['BTC资金曲线'] = (all_equity_df['BTC涨跌幅'] + 1).cumprod()

# 生成画图数据字典，可以画出所有offset资金曲线以及各个offset资金曲线
data_dict = {'多空资金曲线': '多空资金曲线', 'BTC资金曲线': 'BTC资金曲线'}
# 对各个offset的表现，按照净值进行从大到小排序
sorted_offset = sorted(sorted_offset.items(), key=lambda d: d[1], reverse=True)
# 遍历所有offset字段，将各个offset资金曲线的数据进行配置，方便后面画图
for col, v in sorted_offset:
    data_dict['offset_' + col.split('_')[1]] = col
pic_title = 'factor:%s_nv:%s_pro:%s_risk:%s' % (factor_list, rtn.at['累积净值', 0], rtn.at['年化收益', 0], rtn.at['最大回撤', 0])
# 调用画图函数
draw_equity_curve_plotly(all_equity_df, data_dict=data_dict, date_col='candle_begin_time', right_axis={'多空最大回撤': '多空dd2here'},
                         title=pic_title)
