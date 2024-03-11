# -*- coding: utf-8 -*-
"""
中性策略框架 | 邢不行 | 2024分享会
author: 邢不行
微信: xbx6660
"""
import sys
from joblib import Parallel, delayed
from Functions import *
from Evaluate import *
from tqdm import tqdm
from Config import *

# 动态读取config里面配置的strategy_name脚本
Strategy = __import__('strategy.%s' % strategy_name, fromlist=('',))

# 获取当前环境下的python解释器
python_exec = sys.executable


def run(args):
    _hold = args[0]
    _offset = args[1]
    str_info = factor_info_to_str(args[2])
    os.system('%s 2_选币_单offset.py %s %s %s ' % (python_exec, _hold, _offset, str_info))
    return


if __name__ == '__main__':
    # =====遍历回测的准备
    # 需要回测的因子
    filter_factor = ['PctChange']
    factor_class_list = [_ for _ in factor_class_list if ('风格' not in _) & (_ not in filter_factor)]  # 排除因子分析工具携带的辅助因子

    # factor_class_list = ['NetTaBuyStd']  # 可以自定义因子进行遍历
    # 从Strategy中获取需要回测持仓周期
    hold_period = Strategy.hold_period
    # 构建几个因子的组合方式(数字越大，越耗时)
    factor_count = 1

    all_factor_column_list = []
    for factor in factor_class_list:
        # 加载因子文件
        _cls = __import__('factors.%s' % factor, fromlist=('',))
        # 兼容get_parameter
        if 'get_parameter' in _cls.__dict__:  # 如果存在get_parameter，直接覆盖config中配置的factor_param_list
            param_list = getattr(_cls, 'get_parameter')()
        else:  # 因子里面没有get_parameter函数，默认是用config中配置的factor_param_list
            param_list = factor_param_list.copy()
        # 构建因子列
        for param in param_list:
            all_factor_column_list.append(f'{factor}_{param}')

    # 通过itertools.combinations构建因子的排列组合
    factors = list(itertools.combinations(all_factor_column_list, factor_count))

    # 通过itertools.product构建True，False的全组合
    reverses = list(itertools.product([True, False], repeat=factor_count))

    # 循环遍历因子组合和排序组合，构建完整的回测因子组合(多参数目前默认全是等权)
    factor_para_list = []
    for f in factors:
        for r in reverses:
            factor_dict = {}
            col = []
            for i in range(0, factor_count):
                factor_dict[f[i]] = r[i]
                col.append(f[i].split('_')[0])  # 如果因子名称中带有下划线，会出现统计错误
            # 去除单因子自己与自己的组合
            if len(set(col)) < factor_count:
                continue
            factor_para_list.append(factor_dict)

    # 构建遍历信息
    ergodic_list = []
    for factor_para in factor_para_list:
        # 设置offset遍历逻辑，默认全offset遍历
        for ofs in list(range(0, int(hold_period[:-1]))):  # 这里可以自定义遍历的offset
            ergodic_list.append([hold_period, ofs, factor_para])

    # ergodic_list = ergodic_list[:10]  # 测试使用
    # =====并行或串行，依次调用2号脚本
    multiply_process = True  # 是否并行。在测试的时候可以改成False，实际跑的时候改成True
    if multiply_process:
        df_list = Parallel(n_jobs=n_jobs)(
            delayed(run)(factor_para)
            for factor_para in tqdm(ergodic_list)
        )
    else:
        for factor_para in tqdm(ergodic_list):
            run(factor_para)
