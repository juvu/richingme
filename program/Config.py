# -*- coding: utf-8 -*-
"""
中性策略框架 | 邢不行 | 2024分享会
author: 邢不行
微信: xbx6660
"""
from program.Functions import *

_ = os.path.abspath(os.path.dirname(__file__))  # 返回当前文件路径
root_path = os.path.abspath(os.path.join(_, '..'))  # 返回根目录文件夹

# 现货数据路径
spot_path = 'D:/Data/Coin/spot_binance_1h/'

# 合约数据路径
swap_path = 'D:/Data/Coin/swap_binance_1h/'

# 回测结果数据路径。用于发帖脚本使用
back_test_path = root_path + '/data/回测结果/'

# strategy目录下的策略文件名，配置后让程序动态加载策略文件。例：Strategy_NetTaBuyStd，Strategy_高频，Strategy_基本面
strategy_name = 'Strategy_ILLQStd2'

# 回测信息配置
start_date = '2024-01-01'  # 回测开始时间
end_date = '2024-03-04'  # 回测结束时间
spot_c_rate = 1 / 1000  # 现货手续费，手续费没有选返佣，整体比较严格。现货返佣50%
swap_c_rate = 4 / 10000  # 合约手续费，手续费没有选返佣，整体比较严格。合约返佣30%+
leverage = 1  # 杠杆数。我看哪个赌狗要把这里改成大于1的。高杠杆如梦幻泡影。不要想着一夜暴富，脚踏实地赚自己该赚的钱。
margin_rate = 0.05  # 维持保证金率，净值低于这个比例会爆仓
black_list = ['BTC-USDT', 'ETH-USDT']  # 拉黑名单，永远不会交易。不喜欢的币、异常的币。例：LUNA-USDT, 这里与实盘不太一样，需要有'-'
white_list = []  # 如果不为空，即只交易这些币，只在这些币当中进行选币。例：LUNA-USDT, 这里与实盘不太一样，需要有'-'
min_kline_num = 168  # 最少上市多久，不满该K线根数的币剔除，即剔除刚刚上市的新币。168：标识168个小时，即：7*24
stable_symbol = ['BKRW', 'USDC', 'USDP', 'TUSD', 'BUSD', 'FDUSD', 'DAI', 'EUR', 'GBP', 'USBP', 'SUSD', 'PAXG']  # 稳定币信息，不参与交易的币种
if_all_curves = True  # 是否生成多头、空头曲线。True：生成；False：不生成，只有多空资金曲线
# 特殊现货对应列表
special_symbol_dict = {
    'DODO': 'DODOX',  # DODO现货对应DODOX合约
    'LUNA': 'LUNA2',  # LUNA现货对应LUNA2合约
    '1000SATS': '1000SATS',  # 1000SATS现货对应1000SATS合约
}

# 需要参与计算的因子列表，用于`1_选币数据整理.py`
factor_class_list = get_file_in_folder(root_path + '/program/factors/', file_type='.py', filters=['__init__'],
                                       drop_type=True)
# factor_class_list = ['QuoteVolumeMean', 'PctChange']  # 计算指定因子列表

# 定义一个开始的基准时间，避免周期转换出现问题
benchmark = pd.DataFrame(pd.date_range(start='2017-01-01', end=end_date, freq='1H'))  # 创建2017-01-01至回测结束时间的1H列表
benchmark.rename(columns={0: 'candle_begin_time'}, inplace=True)

# 回测因子的参数列表。
# 因子文件如果有 get_parameter 函数，优先使用 get_parameter。
# 因子文件如果没有 get_parameter 函数，默认使用下面的 factor_param_list。
factor_param_list = [24, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336]

# 全局多进程核心数量控制。建议最大设置：当前cpu最大核心数的一半
# 想知道自己的cpu有多少核心？ os.cpu_count() 可以查看
n_jobs = 14
