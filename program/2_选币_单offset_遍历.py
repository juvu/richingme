# -*- coding: utf-8 -*-
"""
中性策略框架 | 邢不行 | 2024分享会
author: 邢不行
微信: xbx6660
"""
import os
import time
import multiprocessing

from Config import *
from Functions import *
from Evaluate import *
import sys

pd.set_option('display.max_rows', 1000)
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行


if __name__ == '__main__':
    # 动态读取config里面配置的strategy_name脚本
    Strategy = __import__('strategy.%s' % strategy_name, fromlist=('',))

    # =====需要手动设置的参数
    bins = 10  # 分箱数。5 表示设置5分箱，0 表示不进行分箱测试

    # =====从配置中读取的数据
    # 从Strategy中获取需要回测的factor_list
    factor_list = Strategy.factor_list
    # 从Strategy中获取需要回测的filter_list
    filter_list = Strategy.filter_list
    # 从Strategy中获取需要回测持仓周期
    hold_period = Strategy.hold_period
    # 从Strategy中获取offset
    offset = Strategy.offset
    # 从Strategy中获取策略文件名stg_name
    stg_name = Strategy.stg_name
    # 从Strategy中获取是否使用现货if_use_spot
    if_use_spot = Strategy.if_use_spot
    # 从Strategy中获取多空选币数量
    long_select_coin_num = Strategy.long_select_coin_num
    short_select_coin_num = Strategy.short_select_coin_num

    # 检查杠杆配置
    check_leverage(Strategy, if_use_spot, leverage)

    # 手动运行标识。手动运行的时候才会出画图，脚本调用直接跳过
    if_manual = True  # 默认手工运行
    if_save = True
    if len(sys.argv) > 1:
        if_manual = False  # 如果是外部运行，则将手工运行的标志设为false

        select_factor = sys.argv[1]
        filter_factor = sys.argv[2]
        hold_period = sys.argv[3]
        offset = int(sys.argv[4])
        select_factor_num = sys.argv[5]
        if_use_spot = eval(sys.argv[6])
        sort_mode = eval(sys.argv[7])
        result_path = sys.argv[8]

        factor_list = [(select_factor.split('_')[0], sort_mode, int(select_factor.split('_')[1]), 1)]
        if filter_factor != 'None':
            filter_list = [(filter_factor.split('_')[0], int(filter_factor.split('_')[1]))]
        else:
            filter_list = []

        if select_factor_num == '0.1':
            long_select_coin_num = 0.1
            short_select_coin_num = 'long_nums'
        else:
            long_select_coin_num = short_select_coin_num = float(select_factor_num)

        if_save = False

    # 判断是否使用现货。如果使用现货，之后保存的文件名中带有SPOT；如果不使用现货，则为SWAP
    if if_use_spot:
        label = 'SPOT'
    else:
        label = 'SWAP'

    # ===加载因子信息
    # =根据配置的因子信息加载因子的数据，例如配置了('ILLQ', True, '7', 1)
    # =factor_column_list列表里面会添加  ILLQ_7 然后加载这个字段
    factor_column_list = []
    for factor_name, if_reverse, parameter_list, weight in factor_list:
        factor_column_list.append(f'{factor_name}_{str(parameter_list)}')

    # =根据配置的过滤信息加载过滤的数据，例如配置了('PctChange', 7)
    # =filter_column_list列表里面会添加  PctChange_7 然后加载这个字段
    filter_column_list = []
    for factor_name, parameter_list in filter_list:
        filter_column_list.append(f'{factor_name}_{str(parameter_list)}')

    # =合并 factor_column_list 和 filter_column_list 的信息（会去重）
    all_factor_list = list(set(factor_column_list + filter_column_list))

    # =打印回测信息
    print('开始回测...')
    print('因子列表：', factor_list)
    print('过滤列表：', filter_list)
    print('持仓周期：', hold_period)
    print('offset：', offset)
    print('是否使用现货：', if_use_spot)

    # ===读取并整理数据
    # =读取之前计算好的pkl数据
    start_time = time.time()
    df = read_coin(root_path, hold_period, all_factor_list, if_use_spot, n_jobs, offset)
    print('读取耗时：', time.time() - start_time)
    if df.empty:
        print('读取df为空，结束当前回测')
        exit()

    # ===将数据分为现货数据和合约数据
    # 合约和现货数据分离
    df_spot = df[df['symbol_type'] == 'spot']  # 取出现货数据，后续根据配置用来进行过滤和选币
    df_swap = df[df['symbol_type'] == 'swap']  # 取出合约数据，后续根据配置用来进行过滤和选币，并替换掉现货中的行情数据

    # 是否使用现货数据进行回测，如果为True，则表明在现货数据中选币；为False，则在合约数据中选币
    if if_use_spot:  # 使用现货数据，则在现货中进行过滤，并选币
        back_test_df = df_spot.copy()
        if back_test_df.empty:  # 如果数据中不存在现货数据就退出
            print('当前为现货模式，但是数据中不存在现货数据，请重新整理数据。')
            exit()
    else:  # 使用现货数据，则在现货中进行过滤，并选币
        back_test_df = df_swap.copy()
    # 删除选币因子为空的数据
    back_test_df.dropna(subset=all_factor_list, inplace=True)

    # ===计算因子
    # 调用Strategy.calc_factor计算最终用来选币的因子，具体的因子计算方法可以在Strategy.calc_factor中自定义
    back_test_df = Strategy.calc_factor(back_test_df, external_list=factor_list)

    # =计算好因子后的一些数据整理工作
    # 获取做多和做空因子的字段名，本框架支持多头和空头是不同的因子
    long_short_columns = list({Strategy.long_factor, Strategy.short_factor})
    # 后期需要保存的列 = 多空因子字段 + 过滤因子字段
    save_columns = list(set(long_short_columns + filter_column_list))
    # 只保留一些有用的字段，减小数据大小
    back_test_df = back_test_df[['candle_begin_time', 'offset', 'symbol', 'symbol_type', 'tag', 'ret_next', '每小时涨跌幅',
                                 '每小时涨跌幅_byclose', 'fundingRate'] + save_columns]  # 只保留需要的字段

    # ===对回测数据进行过滤
    # 过滤的具体方法可以在Strategy.before_filter中自定义，故多空可能是分开过滤的，所以需要多头给一个df，空头也给一个df
    long_df, short_df = Strategy.before_filter(back_test_df, ex_filter_list=filter_list)
    # 如果使用现货数据，多头选币不受影响，但是空头选币只能在有合约的现货中进行选币
    if if_use_spot:
        short_df = short_df[short_df['tag'] == 'HasSwap']  # 保留有合约的现货

    # 如果是手动运行，则生成分箱图
    if if_manual:
        # 分组测试稳定性，多头空头分别生成分箱图
        robustness_test(long_df, short_df, bins=bins, long_factor=Strategy.long_factor, short_factor=Strategy.short_factor)

    # ===选币操作
    # 根据选币数量进行多空选币操作
    select_coin = select_long_and_short_coin(long_df, short_df, long_select_coin_num, short_select_coin_num,
                                             long_factor=Strategy.long_factor, short_factor=Strategy.short_factor)
    # 如果是现货模式，将现货涨跌幅数据换成合约涨跌幅数据
    if if_use_spot:
        select_coin = transfer_swap(select_coin, df_swap, special_symbol_dict)

    # ===计算资金曲线前的一些操作
    # 针对合约和现货给予不同的手续费率
    select_coin.loc[select_coin['symbol_type'] == 'spot', 'rate'] = spot_c_rate
    select_coin.loc[select_coin['symbol_type'] == 'swap', 'rate'] = swap_c_rate
    select_coin['lvg'] = leverage

    # 保留指定字段
    select_coin = select_coin[['candle_begin_time', 'offset', 'symbol', 'symbol_type', '方向', '每小时涨跌幅', '每小时涨跌幅_byclose', 'fundingRate', 'rate', 'lvg', 'ret_next']]

    # 在选币信息上带上symbol和方向的信息，例如 做空BTC-USDT 会变成 BTC-USDT(-1)
    select_coin['选币'] = select_coin['symbol'] + '(' + select_coin['symbol_type'] + ',' + select_coin['方向'].astype(str) + ')' + ' '

    # 把选币结果重新排序以及重置索引
    select_coin.sort_values(by='candle_begin_time', inplace=True)
    select_coin.reset_index(drop=True, inplace=True)

    # 输出选币结果（主要是给中性策略查看器用的）
    if if_save:
        save_path = os.path.join(back_test_path, f'{stg_name}_{label}_选币结果_{hold_period}_{offset}.csv')
        select_coin.to_csv(save_path, index=False, encoding='gbk')
        # print(select_coin)

    print('选币完成：', time.time() - start_time)

    # ===计算资金曲线
    # =构建回测期间，完整的交易时间列表
    trading_time_list = create_trading_time(benchmark, hold_period, offset, start_date, end_date)

    # =将多头和空头拆开计算
    long_df = select_coin[select_coin['方向'] == 1]
    short_df = select_coin[select_coin['方向'] == -1]

    # 调用函数计算扣除资金费 & 手续费后的每日净值
    # long_df = cal_net_value(long_df, hold_period, trading_time_list)
    # short_df = cal_net_value(short_df, hold_period, trading_time_list)
    pool = multiprocessing.Pool(processes=3)
    result1 = pool.apply_async(cal_net_value, (long_df, hold_period, trading_time_list))
    result2 = pool.apply_async(cal_net_value, (short_df, hold_period, trading_time_list))
    result3 = pool.apply_async(calc_swap_pos_net_value,
                               (select_coin, leverage, hold_period, margin_rate, trading_time_list))
    pool.close()
    long_df = result1.get()
    short_df = result2.get()

    # =计算合约仓位信息
    # PS: 放在这里计算，是因为保留最原始的涨跌幅数据，我们需要单独计算合约仓位的净值
    # swap_net_value = calc_swap_pos_net_value(select_coin, leverage, hold_period, margin_rate, trading_time_list)
    swap_net_value = result3.get()
    print('计算多空资金曲线 + 合约仓位资金曲线：', time.time() - start_time)

    # 创造空的时间周期表，用于填充不选币的周期
    empty_df = create_empty_data(benchmark, hold_period, offset=offset)
    # 填充缺失数据
    empty_df.update(swap_net_value)
    swap_net_value = empty_df.copy()
    swap_net_value['选币'].fillna(method='ffill', inplace=True)
    swap_net_value.dropna(subset=['选币'], inplace=True)
    swap_net_value['每小时资金曲线'] = swap_net_value['每小时资金曲线'].apply(list)

    # 将多空方向处理后的数据合并起来
    select_coin = pd.concat([long_df, short_df], ignore_index=True)
    print('多空资金曲线：', time.time() - start_time)

    # ===计算当周期资金曲线
    # 将多空选币的资金曲线合并成当周期内的资金曲线
    group = select_coin.groupby('candle_begin_time')
    # 存储周期内的资金曲线
    merge_df = pd.DataFrame()
    merge_df['选币'] = group['选币'].sum()
    merge_df['offset'] = group['offset'].last()
    # 将多空方向的资金曲线合并起来
    merge_df['每小时资金曲线'] = group['每小时资金曲线'].apply(lambda x: np.array(x).mean(axis=0))
    # 计算周期的涨跌幅
    merge_df['周期涨跌幅'] = merge_df['每小时资金曲线'].apply(lambda x: x[-1] - 1)
    # 通过每小时资金曲线，计算每小时账户资金的涨跌幅
    merge_df['每小时涨跌幅'] = merge_df['每小时资金曲线'].apply(lambda x: list(pd.DataFrame([1] + list(x)).pct_change()[0].iloc[1:]))
    # 计算整个周期多空的平均调仓
    merge_df['多空调仓比例'] = group['调仓比例'].mean()

    # 将以close计算的每小时涨跌幅(用来轮动)也做相同的操作
    merge_df['每小时资金曲线_byclose'] = group['每小时资金曲线_byclose'].apply(lambda x: np.array(x).mean(axis=0))
    merge_df['周期涨跌幅_byclose'] = merge_df['每小时资金曲线_byclose'].apply(lambda x: x[-1] - 1)
    merge_df['每小时涨跌幅_byclose'] = merge_df['每小时资金曲线_byclose'].apply(lambda x: list(pd.DataFrame([1] + list(x)).pct_change()[0].iloc[1:]))

    # 判断是否生成多头、空头的资金曲线。
    if if_all_curves:
        # 计算多空方向的资金曲线
        if long_df.empty:
            merge_df['多头每小时资金曲线'] = None
            merge_df['多头周期涨跌幅'] = None
            merge_df['多头每小时涨跌幅'] = None
            merge_df['多头调仓比例'] = None

            # 以close计算的涨跌幅也做相同的操作
            merge_df['多头每小时资金曲线_byclose'] = None
            merge_df['多头周期涨跌幅_byclose'] = None
            merge_df['多头每小时涨跌幅_byclose'] = None
        else:
            long_df = long_df.set_index('candle_begin_time')
            merge_df['多头每小时资金曲线'] = long_df.loc[:, '每小时资金曲线']
            merge_df['多头周期涨跌幅'] = merge_df['多头每小时资金曲线'].apply(lambda x: x[-1] - 1)
            merge_df['多头每小时涨跌幅'] = merge_df['多头每小时资金曲线'].apply(lambda x: list(pd.DataFrame([1] + list(x)).pct_change()[0].iloc[1:]))
            merge_df['多头调仓比例'] = long_df.loc[:, '调仓比例']

            # 同上
            merge_df['多头每小时资金曲线_byclose'] = long_df.loc[:, '每小时资金曲线_byclose']
            merge_df['多头周期涨跌幅_byclose'] = merge_df['多头每小时资金曲线_byclose'].apply(lambda x: x[-1] - 1)
            merge_df['多头每小时涨跌幅_byclose'] = merge_df['多头每小时资金曲线_byclose'].apply(lambda x: list(pd.DataFrame([1] + list(x)).pct_change()[0].iloc[1:]))

        if short_df.empty:
            merge_df['空头每小时资金曲线'] = None
            merge_df['空头周期涨跌幅'] = None
            merge_df['空头每小时涨跌幅'] = None
            merge_df['空头调仓比例'] = None

            # 同上
            merge_df['空头每小时资金曲线_byclose'] = None
            merge_df['空头周期涨跌幅_byclose'] = None
            merge_df['空头每小时涨跌幅_byclose'] = None
        else:
            short_df = short_df.set_index('candle_begin_time')
            merge_df['空头每小时资金曲线'] = short_df.loc[:, '每小时资金曲线']
            merge_df['空头周期涨跌幅'] = merge_df['空头每小时资金曲线'].apply(lambda x: x[-1] - 1)  # 计算周期的涨跌幅
            merge_df['空头每小时涨跌幅'] = merge_df['空头每小时资金曲线'].apply(lambda x: list(pd.DataFrame([1] + list(x)).pct_change()[0].iloc[1:]))  # 通过每小时资金曲线，计算每小时账户资金的涨跌幅
            merge_df['空头调仓比例'] = short_df.loc[:, '调仓比例']

            # 同上
            merge_df['空头每小时资金曲线_byclose'] = short_df.loc[:, '每小时资金曲线_byclose']
            merge_df['空头周期涨跌幅_byclose'] = merge_df['空头每小时资金曲线_byclose'].apply(lambda x: x[-1] - 1)
            merge_df['空头每小时涨跌幅_byclose'] = merge_df['空头每小时资金曲线_byclose'].apply(lambda x: list(pd.DataFrame([1] + list(x)).pct_change()[0].iloc[1:]))

    # =====计算小时级别的资金曲线
    # 创造空的时间周期表，用于填充不选币的周期
    empty_df = create_empty_data(benchmark, hold_period, offset=offset)
    # 填充缺失数据
    empty_df.update(merge_df)
    merge_df = empty_df.copy()
    merge_df['选币'].fillna(method='ffill', inplace=True)
    merge_df.dropna(subset=['选币'], inplace=True)
    print('填充当周期资金曲线：', time.time() - start_time)

    # 将合并后的资金曲线，与benchmark合并
    if if_all_curves:  # 如果生成多头、空头的资金曲线，在merge时多加两列
        equity = pd.merge(left=benchmark, right=merge_df[['选币', '多空调仓比例', '多头调仓比例', '空头调仓比例']], on=['candle_begin_time'], how='left', sort=True)
    else:
        equity = pd.merge(left=benchmark, right=merge_df[['选币', '多空调仓比例']], on=['candle_begin_time'], how='left', sort=True)
    # 填充选币数据
    equity['选币'].fillna(method='ffill', inplace=True)
    equity.dropna(subset=['选币'], inplace=True)
    # 将每小时涨跌幅数据，填充到index中
    equity['涨跌幅'] = revise_data_length(merge_df['每小时涨跌幅'].sum(), len(equity))
    equity['涨跌幅'].fillna(value=0, inplace=True)
    # 计算最终每小时净值变化
    equity['多空资金曲线'] = (equity['涨跌幅'] + 1).cumprod()
    equity['多空资金曲线'].fillna(value=1, inplace=True)
    # 计算爆仓
    if not swap_net_value.empty:
        equity['合约资金曲线'] = revise_data_length(swap_net_value['每小时资金曲线'].sum(), len(equity), value=1)
        equity.loc[equity['合约资金曲线'] == 0, '是否爆仓'] = 1
        equity.loc[equity['多空资金曲线'] < margin_rate, '是否爆仓'] = 1
    else:
        equity['是否爆仓'] = 0
    # 这里计算合约仓位爆仓之后，直接将现货仓位也同步归零，实际实盘的时候的会留有部分现货仓位的底仓
    equity['是否爆仓'].fillna(method='ffill', inplace=True)
    equity.loc[equity['是否爆仓'] == 1, '多空资金曲线'] = 0
    equity.loc[equity['是否爆仓'] == 1, '涨跌幅'] = 0
    equity.loc[equity['是否爆仓'] == 1, '多空调仓比例'] = 0
    # 保存资金曲线文件
    equity['本周期多空涨跌幅'] = equity['涨跌幅']

    # 同上
    equity['涨跌幅_byclose'] = revise_data_length(merge_df['每小时涨跌幅_byclose'].sum(), len(equity))
    equity['涨跌幅_byclose'].fillna(value=0, inplace=True)
    equity['多空资金曲线_byclose'] = (equity['涨跌幅_byclose'] + 1).cumprod()
    equity['多空资金曲线_byclose'].fillna(value=1, inplace=True)
    equity.loc[equity['是否爆仓'] == 1, '多空资金曲线_byclose'] = 0
    equity.loc[equity['是否爆仓'] == 1, '涨跌幅_byclose'] = 0

    # 计算多空策略的评价指标
    rtn, year_return, month_return = strategy_evaluate(equity, net_col='多空资金曲线', pct_col='本周期多空涨跌幅', turnover_col='多空调仓比例')

    # 如果生乘多头、空头的资金曲线
    if if_all_curves:
        # 合并多头、空头每小时涨跌幅 并 计算资金曲线
        equity['多头涨跌幅'] = revise_data_length(merge_df['多头每小时涨跌幅'].sum(), len(equity))
        equity['多头涨跌幅'].fillna(value=0, inplace=True)
        equity['空头涨跌幅'] = revise_data_length(merge_df['空头每小时涨跌幅'].sum(), len(equity))
        equity['空头涨跌幅'].fillna(value=0, inplace=True)
        equity['多头资金曲线'] = (equity['多头涨跌幅'] + 1).cumprod()
        equity['多头资金曲线'].fillna(value=1, inplace=True)
        equity['空头资金曲线'] = (equity['空头涨跌幅'] + 1).cumprod()
        equity['空头资金曲线'].fillna(value=1, inplace=True)

        # 同上
        equity['多头涨跌幅_byclose'] = revise_data_length(merge_df['多头每小时涨跌幅_byclose'].sum(), len(equity))
        equity['多头涨跌幅_byclose'].fillna(value=0, inplace=True)
        equity['多头资金曲线_byclose'] = (equity['多头涨跌幅_byclose'] + 1).cumprod()
        equity['多头资金曲线_byclose'].fillna(value=1, inplace=True)
        equity['空头涨跌幅_byclose'] = revise_data_length(merge_df['空头每小时涨跌幅_byclose'].sum(), len(equity))
        equity['空头涨跌幅_byclose'].fillna(value=0, inplace=True)
        equity['空头资金曲线_byclose'] = (equity['空头涨跌幅_byclose'] + 1).cumprod()
        equity['空头资金曲线_byclose'].fillna(value=1, inplace=True)

        # =====策略评价
        # 计算纯多头
        long_rtn, long_year_return, long_month_return = strategy_evaluate(equity, net_col='多头资金曲线', pct_col='多头涨跌幅', turnover_col='多头调仓比例')
        # 计算纯空头
        short_rtn, short_year_return, short_month_return = strategy_evaluate(equity, net_col='空头资金曲线', pct_col='空头涨跌幅', turnover_col='空头调仓比例')

        # 多空策略评价合并
        rtn_all = pd.concat([rtn.T, long_rtn.T, short_rtn.T], ignore_index=True)
        rtn_all = rtn_all.T.rename(columns={0: '中性', 1: '多头', 2: '空头'})
        print('\n\n', rtn_all)
        # 多空分年收益率
        year_return_all = pd.concat([year_return.T, long_year_return.T, short_year_return.T], ignore_index=True)
        year_return_all = year_return_all.T.rename(columns={0: '中性', 1: '多头', 2: '空头'})
        print('\n\n分年收益率：\n', year_return_all)
        # 分月双分组
        print('\n\n多头分月总收益率：\n', long_month_return, '\n\n空头分月总收益率：\n', short_month_return, '\n\n中性分月总收益率：\n', month_return)
    else:
        print('策略评价：', time.time() - start_time)
        print(rtn, '\n\n分年收益率：\n', year_return, '\n\n分月收益率：\n', month_return)

    if if_save:
        # 保存资金曲线
        save_path = os.path.join(back_test_path, f'{stg_name}_{label}_资金曲线_{hold_period}_{offset}.csv')
        equity.to_csv(save_path, encoding='gbk', index=False)

    # =====画图
    if if_manual:
        BTC = pd.read_csv(swap_path + 'BTC-USDT.csv', encoding='gbk', parse_dates=['candle_begin_time'], skiprows=1)
        BTC['BTC涨跌幅'] = BTC['close'].pct_change()
        equity = pd.merge(left=equity, right=BTC[['candle_begin_time', 'BTC涨跌幅']], on=['candle_begin_time'], how='left')
        equity['BTC涨跌幅'].fillna(value=0, inplace=True)
        equity['BTC资金曲线'] = (equity['BTC涨跌幅'] + 1).cumprod()

        ETH = pd.read_csv(swap_path + 'ETH-USDT.csv', encoding='gbk', parse_dates=['candle_begin_time'], skiprows=1)
        ETH['ETH涨跌幅'] = ETH['close'].pct_change()
        equity = pd.merge(left=equity, right=ETH[['candle_begin_time', 'ETH涨跌幅']], on=['candle_begin_time'], how='left')
        equity['ETH涨跌幅'].fillna(value=0, inplace=True)
        equity['ETH资金曲线'] = (equity['ETH涨跌幅'] + 1).cumprod()

        # 生成画图数据字典，可以画出所有offset资金曲线以及各个offset资金曲线
        data_dict = {'多空资金曲线': '多空资金曲线', 'BTC资金曲线': 'BTC资金曲线', 'ETH资金曲线': 'ETH资金曲线'}
        right_axis = {'多空最大回撤': '多空dd2here'}

        # 如果画多头、空头资金曲线，同时也会画上回撤曲线
        if if_all_curves:
            data_dict['多头资金曲线'] = '多头资金曲线'
            data_dict['空头资金曲线'] = '空头资金曲线'
            right_axis['多头最大回撤'] = '多头dd2here'
            right_axis['空头最大回撤'] = '空头dd2here'
        pic_title = 'factor:%s_nv:%s_pro:%s_risk:%s_filter:%s' % (
            factor_list, rtn.at['累积净值', 0], rtn.at['年化收益', 0], rtn.at['最大回撤', 0], filter_list)
        # 调用画图函数
        draw_equity_curve_plotly(equity, data_dict=data_dict, date_col='candle_begin_time', right_axis=right_axis,
                                 title=pic_title)
    else:
        # 5、记录回测的结果
        stg_res = pd.DataFrame()
        stg_res.loc[0, '选币因子'] = str(factor_list)
        stg_res.loc[0, '过滤因子'] = str(filter_list)
        stg_res.loc[0, '持仓周期'] = hold_period
        stg_res.loc[0, 'offset'] = offset
        stg_res.loc[0, '选币数量'] = select_factor_num
        stg_res.loc[0, '是否使用现货'] = if_use_spot
        year_return['涨跌幅'] = year_return['涨跌幅'].map(lambda x: float(x[:-1]))
        stg_res['各年收益'] = [year_return.values.reshape(-1).tolist()]
        # 记录回测的各项指标
        stg_res = pd.concat([stg_res, rtn.T], axis=1)
        # 保存文件
        if os.path.exists(result_path):  # 如果文件存在，往原有的文件中添加新的结果
            stg_res.to_csv(result_path, encoding='gbk', index=False, header=False, mode='a')
        else:  # 如果文不件存在，常规的to_csv操作
            stg_res.to_csv(result_path, encoding='gbk', index=False)
