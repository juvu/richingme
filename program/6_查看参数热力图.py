# -*- coding: utf-8 -*-
"""
中性策略框架 | 邢不行 | 2024分享会
author: 邢不行
微信: xbx6660
"""
import os
import sys
import pandas as pd
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from Config import strategy_name, back_test_path, n_jobs
import seaborn as sns

# 动态读取config里面配置的strategy_name脚本
Strategy = __import__('strategy.%s' % strategy_name, fromlist=('',))

# 获取当前环境下的python解释器
python_exec = sys.executable


def run(args):
    _select_factor = args[0]
    _filter_factor = args[1]
    _hold_period = args[2]
    _offset = args[3]
    _select_coin_num = args[4]
    _if_use_spot = args[5]
    _sort_mode = args[6]
    _result_path = args[7]

    os.system('%s 2_选币_单offset_遍历.py %s %s %s %s %s %s %s %s' % (python_exec, _select_factor, _filter_factor, _hold_period, _offset, _select_coin_num, _if_use_spot, _sort_mode, _result_path))
    return


if __name__ == '__main__':
    # 运行前需要将因子对应的参数整理好
    # 运行前需要把之前遍历保存下来的数据删掉，或者改个文件名

    # =====遍历回测的准备
    hold_period = '1D'  # 指定hold_period
    offset = 0  # 指定offset
    select_coin_num = 0.1  # 指定选币数量。当选币数量设置为0.1时，空头数量默认为long_nums
    if_use_spot = True  # 是否使用现货
    result_path = os.path.join(back_test_path + '回测结果汇总_参数热力图.csv')

    # 显示的指标
    indicator = '累积净值'  # 可以设置累积净值、年化收益、最大回撤、年化收益回撤比中的任意一个

    # 选币因子的配置--纵轴
    select_factors = 'QuoteVolumeMean'  # 设置选币因子
    select_params = [3, 7, 14, 30]  # 设置选币因子的参数。设置前应该保证对应参数数据已经被整理过
    sort_mode = True  # 默认为True

    # 过滤因子的配置--横轴
    filter_factors = 'PctChange'  # 设置过滤因子。默认过滤参数为7
    filter_params = [3, 7, 14, 30]  # 设置过滤因子的参数。设置前应该保证对应参数数据已经被整理过

    # 将选币因子和参数组合一起
    select_factor_list = []
    for param in select_params:
        select_factor = select_factors + '_' + str(param)
        select_factor_list.append(select_factor)
    select_factor_list = list(set(select_factor_list))

    # 将选币因子和参数组合一起
    filter_factor_list = []
    for param in filter_params:
        filter_factor = filter_factors + '_' + str(param)
        filter_factor_list.append(filter_factor)
    filter_factor_list = list(set(filter_factor_list))

    # 将各个参数组合一下，每个info是遍历一次传入的所有参数
    infos = []
    for select_factor in select_factor_list:
        for filter_factor in filter_factor_list:
            info = [select_factor, filter_factor, hold_period, offset, select_coin_num, if_use_spot, sort_mode, result_path]
            infos.append(info)

    # =====并行或串行，依次调用2号脚本
    multiply_process = True  # 是否并行。在测试的时候可以改成False，实际跑的时候改成True
    if multiply_process:
        df_list = Parallel(n_jobs=n_jobs)(delayed(run)(info) for info in tqdm(infos))
    else:
        for info in tqdm(infos):
            run(info)

    # 读取保存的数据
    if not os.path.exists(result_path):
        print(f'参数热力图计算统计结果不存在，请检查当前遍历配置信息是否正确 或【{back_test_path}】目录下是否存在文件')
        exit()
    result = pd.read_csv(result_path, encoding='gbk')
    result = result.drop_duplicates().reset_index(drop=True)
    result = result[(result['持仓周期'] == hold_period) & (result['offset'] == offset) & (result['选币数量'] == select_coin_num) &
                    (result['是否使用现货'] == if_use_spot)]
    result['选币参数'] = result['选币因子'].apply(lambda x: int(x.split(',')[2].strip()))
    result['选币因子'] = result['选币因子'].apply(lambda x: x.split("'")[1])
    result['过滤参数'] = result['过滤因子'].apply(lambda x: int(x.split(',')[-1].split(')')[0].strip()))
    result['过滤因子'] = result['过滤因子'].apply(lambda x: x.split("'")[1])

    # 转换数据格式
    result['累积净值'] = result['累积净值'].map(lambda x: float(x))
    result['年化收益'] = result['年化收益'].map(lambda x: float(x[:-1]))
    result['最大回撤'] = result['最大回撤'].map(lambda x: float(x[:-1]))
    result['年化收益/回撤比'] = result['年化收益/回撤比'].map(lambda x: float(x))

    # 画参数热力图
    temp = pd.pivot_table(result, values=indicator, index=result['选币参数'], columns=result['过滤参数'])

    sns.set()  # 设置一下展示的主题和样式
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.title(f'热力图 {indicator}')  # 设置标题
    sns.heatmap(temp, annot=True, xticklabels=temp.columns, yticklabels=temp.index, fmt=".2g")  # 画图
    plt.show()
