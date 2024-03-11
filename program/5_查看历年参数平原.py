# -*- coding: utf-8 -*-
"""
中性策略框架 | 邢不行 | 2024分享会
author: 邢不行
微信: xbx6660
"""
import os
import sys
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from tqdm import tqdm
import ast
from Config import strategy_name, back_test_path, start_date, end_date, n_jobs
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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
    result_path = os.path.join(back_test_path + '回测结果汇总_历年参数平原.csv')

    # 选币因子的配置
    select_factors = 'QuoteVolumeMean'  # 设置选币因子
    select_params = [3, 7, 14, 30]  # 设置选币因子的参数。设置前应该保证对应参数数据已经被整理过
    sort_mode = True  # 默认为True

    # 过滤因子的配置
    filter_factors = ''  # 设置过滤因子。默认过滤参数为7。支持过滤因子为空
    keep_same_params = True  # 默认过滤因子参数为7，如果想让过滤因子的参数和选币因子的参数保持一致，设置为True

    # 将选币因子和参数组合一起
    select_factor_list = []
    for param in select_params:
        select_factor = select_factors + '_' + str(param)
        select_factor_list.append(select_factor)
    select_factor_list = list(set(select_factor_list))

    # 将各个参数组合一下，每个info是遍历一次传入的所有参数
    infos = []
    for select_factor in select_factor_list:
        if filter_factors != '':
            if keep_same_params:
                filter_factor = filter_factors.split('_')[0] + '_' + select_factor.split('_')[1]
            else:
                filter_factor = filter_factors
        else:
            filter_factor = None

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
        print(f'参数平原计算统计结果不存在，请检查当前遍历配置信息是否正确 或【{back_test_path}】目录下是否存在文件')
        exit()
    result = pd.read_csv(result_path, encoding='gbk')
    result = result.drop_duplicates().reset_index(drop=True)
    result = result[(result['持仓周期'] == hold_period) & (result['offset'] == offset) & (result['选币数量'] == select_coin_num) &
                    (result['是否使用现货'] == if_use_spot)]
    result['参数'] = result['选币因子'].apply(lambda x: int(x.split(',')[2].strip()))
    result['选币因子'] = result['选币因子'].apply(lambda x: x.split("'")[1])
    if '[]' not in result['过滤因子'].values.tolist():
        result['过滤因子'] = result['过滤因子'].apply(lambda x: x.split("'")[1])
    else:
        result['过滤因子'] = ''

    # 转换数据格式
    result['累积净值'] = result['累积净值'].map(lambda x: float(x))
    years = list(range(int(start_date.split('-')[0]), int(end_date.split('-')[0]) + 1, 1))
    result['各年收益'] = result['各年收益'].apply(ast.literal_eval)
    for i, year in enumerate(years):
        result[year] = result['各年收益'].map(lambda x: (x[i]) / 100 + 1)

    # 画历年参数平原图
    fig, axs = plt.subplots(nrows=len(years) + 1, ncols=1)
    xticks = result['参数'].to_list()
    x_tick_labels = [f'{t}' for t in xticks]
    axs[0].bar(result['参数'], result['累积净值'], width=0.5)
    axs[0].set_title('累计净值')
    axs[0].set_xticks(xticks)
    axs[0].set_xticklabels(x_tick_labels)
    for index, year in enumerate(years):
        axs[index + 1].bar(result['参数'], result[year], width=0.5)
        axs[index + 1].set_title(f'{year}')
        axs[index + 1].set_xticks(xticks)
        axs[index + 1].set_xticklabels(x_tick_labels)
    plt.suptitle('历年参数平原图')
    plt.tight_layout()
    plt.show()
