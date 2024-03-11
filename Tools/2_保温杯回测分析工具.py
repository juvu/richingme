"""
中性策略框架 | 邢不行 | 2024分享会
author: 邢不行
微信: xbx6660
"""
import pandas as pd

from Tools.utils.sFunction import *
from program.Config import *

warnings.filterwarnings('ignore')

_ = os.path.abspath(os.path.dirname(__file__))  # 返回当前文件路径
root_path = os.path.abspath(os.path.join(_, '../'))  # 返回根目录文件夹

# =====需要配置的参数
start = '2023-02-21'  # 分析开始时间
end = '2023-07-10'  # 分析结束时间
filename = "QuoteVolumeMean_SPOT_选币结果_1D_0.csv"

# 主图增加,均为折线图。
# 使用ma时，需要确保因子库中存在对于的因子数据。
add_factor_main_list = [
    {'因子名称': 'btc', '次坐标轴': True},
    {'因子名称': 'ma_7', '次坐标轴': False, 'period': 'H'},
    {'因子名称': 'ma_8', '次坐标轴': False, 'period': 'H'},
]

# 附图增加，一个dict为一个子图
# 因子名称的list大于1个值，则会被画在同一个图中，没用次坐标轴概念
# 图形样式有且仅有三种选择K线图\柱状图\折线图
# 对于所有不带'_'的因子名称，都尝试使用小时K载入的数据；对于带有'_'的因子，不指定period参数，即为策略使用的因子period。
add_factor_sub_list = [
    {'因子名称': ['btc'], '图形样式': 'K线图'},
    {'因子名称': ['volume'], '图形样式': '柱状图'},
    {'因子名称': ['PctChange_7'], '图形样式': '折线图'},
]
# 按因子名称指定颜色，K线展示的内容固定颜色指定无效。颜色仅为plotly支持的颜色格式（可通过webcolors.CSS3_NAMES_TO_HEX查询）。
# 因子可不指定颜色，使用彩虹集团的随机配色
color_dict = {'btc': 'royalblue', 'volume': 'yellow'}

name = filename.split('_')[0]  # 策略名称
select_path = root_path + f"/data/回测结果/{filename}"
strategy_name = f'Strategy_{name}'
Strategy = __import__('program.strategy.%s' % strategy_name, fromlist=('',))
hold_period = Strategy.hold_period
period = pd.to_timedelta(hold_period)

# # ===加载因子信息
all_factor_dict = {}
for each in add_factor_main_list:
    if '_' in each['因子名称']:
        if 'period' not in each.keys():
            each['period'] = hold_period[-1].upper()
        key = each['period']
    else:
        key = 'SPECIAL'
    if key not in all_factor_dict.keys():
        all_factor_dict[key] = []
    all_factor_dict[key].append(each['因子名称'])

for each_sub in add_factor_sub_list:
    for each in each_sub['因子名称']:
        if '_' in each:
            if 'period' not in each_sub.keys():
                each_sub['period'] = hold_period[-1].upper()
            key = each_sub['period']
        else:
            key = 'SPECIAL'
        if key not in all_factor_dict.keys():
            all_factor_dict[key] = []
        all_factor_dict[key].append(each)
for p in all_factor_dict.keys():
    all_factor_dict[p] = list(set(all_factor_dict[p]))

if 'btc' in all_factor_dict['SPECIAL']:
    btc = pd.read_csv(swap_path + 'BTC-USDT.csv', encoding='gbk', parse_dates=['candle_begin_time'], skiprows=1)
    btc.rename(columns={key: f'btc_{key}' for key in ['open', 'high', 'low', 'close']}, inplace=True)
    btc = btc[['candle_begin_time'] + [f'btc_{key}' for key in ['open', 'high', 'low', 'close']]]
print(all_factor_dict)
# =====数据的处理及周期处理
# ===读取选币数据 & 截取数据
# 读取选币数据
select = pd.read_csv(select_path, encoding='gbk', parse_dates=['candle_begin_time'])
# select = select[select['symbol_type'] == 'swap']
# 只保留分析区间内的选币数据
select = select[select['candle_begin_time'] >= pd.to_datetime(start)]
select = select[select['candle_begin_time'] <= pd.to_datetime(end)]

# 计算一天有多少个period，为之后算日化收益做准备
daily_periods = pd.to_timedelta('1D') / period

# 计算这个区间内有多个个period，为之后画图做准备
total_periods = (pd.to_datetime(end) - pd.to_datetime(start)) / period

# 统一把卖出时间算好，是candle_begin_time按照period推下一个周期的上一个小时的结束。（因为画图是小时图）
select['end_time'] = select['candle_begin_time'] + pd.to_timedelta(period) - pd.to_timedelta('1H')

# =====按照币种及方向进行分组，计算一些分析所需要的数据
groups = select.groupby(['symbol', 'symbol_type', '方向'])
res_list = []  # 储存分组结果的list
# 遍历各个分组
for t, g in groups:
    # ===分组处理数据
    g.sort_values(by='candle_begin_time', inplace=True)
    # 计算前后K线的时间间隔
    g['span'] = g['candle_begin_time'].diff()
    # 时间间隔不等于设定周期的肯定是中间有过换仓，标记一下换仓的开始时间
    g.loc[g['span'] != period, 'start_time'] = g['candle_begin_time']
    g['start_time'].fillna(method='ffill', inplace=True)
    # 按照不同的持仓时间段分类，并计算持仓列表
    hold = g.groupby('start_time').agg({'end_time': 'last'}).reset_index()

    hold['hold_info'] = hold['start_time'].apply(str) + '--' + hold['end_time'].apply(str)
    # 结合持仓数据，计算真实的收益（忽略手续费）
    g['ret_next_real'] = g['ret_next'] * t[2]

    # ===统计结果
    res = pd.DataFrame()  # 需要返回的结果
    res.loc[0, 'symbol'] = t[0]  # 币种名称
    res.loc[0, 'symbol_type'] = t[1]  # 现货or合约
    res.loc[0, '方向'] = t[2]  # 交易方向
    res.loc[0, '选中次数'] = len(g['start_time'].unique())  # 区间内选中币种的次数
    res.loc[0, '累计周期数'] = g.shape[0]  # 区间内币种的持仓周期数
    res.loc[0, '每周期平均收益（日化）'] = (g['ret_next_real'].mean() + 1) ** daily_periods - 1  # 不同周期的平均日化收益
    # 区间内，币种的累计收益（准确的说是净值的百分比）
    if t[2] == 1:
        res.loc[0, '区间累计收益'] = (g['ret_next'] + 1).prod() - 1  # 做多的最终收益
    else:
        res.loc[0, '区间累计收益'] = 1 - (g['ret_next'] + 1).prod()  # 做空的最终收益
    res.loc[0, '首次选中时间'] = sorted(g['start_time'].unique())[0]  # 区间内首次选中币种的时间
    res.loc[0, '最后选中时间'] = sorted(g['start_time'].unique())[-1]  # 区间内最后一次选中币种的时间

    # 计算区间内的的持币时间段，格式如：['2021-05-09 13:00:00--2021-05-09 15:00:00', '2021-05-10 19:00:00--2021-05-11 05:00:00']
    res['选中周期'] = ''  # 小tips：需要往DataFrame的cell里面插入list，这一列需要是object类型（所以这里给了''，字符串就是object）
    res.at[0, '选中周期'] = hold['hold_info'].to_list()  # 往数据中插入list时，需要用at函数，loc不行。

    # 将计算的结果添加到结果汇总中
    res_list.append(res)

# =====汇总所有结果，再按照方向拆成多头和空头
# 汇总所有分组的分析结果
all_res = pd.concat(res_list, ignore_index=True)
# 拆分多头的分析结果
long_res = all_res[all_res['方向'] == 1].reset_index(drop=True)
# 拆分空头的分析结果
short_res = all_res[all_res['方向'] == -1].reset_index(drop=True)

# =====针对多空分析结果进行进一步分析
describe = pd.DataFrame()  # 分析结果储存的df

# 1.分析多头数据
describe.loc['选币数', '多头'] = long_res.shape[0]
describe.loc['平均选中次数', '多头'] = long_res['选中次数'].mean()
describe.loc['平均周期数', '多头'] = long_res['累计周期数'].mean()
describe.loc['平均日化收益', '多头'] = long_res['每周期平均收益（日化）'].mean()
describe.loc['平均累计收益', '多头'] = long_res['区间累计收益'].mean()
describe.loc['胜率_日均', '多头'] = long_res[long_res['每周期平均收益（日化）'] > 0].shape[0] / long_res.shape[0]
describe.loc['胜率_累计', '多头'] = long_res[long_res['区间累计收益'] > 0].shape[0] / long_res.shape[0]

# 2.分析空头数据
describe.loc['选币数', '空头'] = short_res.shape[0]
describe.loc['平均选中次数', '空头'] = short_res['选中次数'].mean()
describe.loc['平均周期数', '空头'] = short_res['累计周期数'].mean()
describe.loc['平均日化收益', '空头'] = short_res['每周期平均收益（日化）'].mean()
describe.loc['平均累计收益', '空头'] = short_res['区间累计收益'].mean()
describe.loc['胜率_日均', '空头'] = short_res[short_res['每周期平均收益（日化）'] > 0].shape[0] / short_res.shape[0]
describe.loc['胜率_累计', '空头'] = short_res[short_res['区间累计收益'] > 0].shape[0] / short_res.shape[0]
# 打印分析结果
describe = format_col(describe)
print(describe)

# =====结果保存
# 保存数据的文件夹是否存在
save_path = root_path + f'/Tools/回测分析结果/{name}_{start}_{end}/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
# 保存多头、空头的分析结果数据 及 汇总数据
long_res.to_csv(save_path + '01_多头分析结果.csv', encoding='gbk', index=False)
short_res.to_csv(save_path + '02_空头分析结果.csv', encoding='gbk', index=False)
describe.to_csv(save_path + '03_分析汇总.csv', encoding='gbk')

# =====根据多空结果，进行绘图
# 绘制的时候K线向历史和前后多扩展15%（最少扩展20个周期）
# K线开始时间
k_start = pd.to_datetime(start) - pd.to_timedelta(period) * max(int(total_periods * 0.15), 20)  # 周期数据开始时间
d_start = pd.to_datetime(start) - pd.to_timedelta('90D')  # 日线数据开始时间
# K线结束时间
k_end = pd.to_datetime(end) + pd.to_timedelta(period) * max(int(total_periods * 0.15), 20)  # 周期数据结束时间
d_end = pd.to_datetime(end) + pd.to_timedelta('90D')  # 日线数据结束时间

benchmark = pd.DataFrame(pd.date_range(start=k_start, end=k_end, freq='1H'))
benchmark.rename(columns={0: 'candle_begin_time'}, inplace=True)

# 画图需要的信息
draw_info = {'多头': long_res, '空头': short_res}


def each_coin_graph_creator(i):
    coin_symbol = info_df.loc[i, 'symbol']
    symbol_type = info_df.loc[i, 'symbol_type']
    print(f'正在绘制：{direction}_{coin_symbol}_{symbol_type}')
    res_loc = info_df.loc[i]
    # 读取币种信息
    k_path = {'swap': swap_path, 'spot': spot_path}[symbol_type]
    df = pd.read_csv(k_path + coin_symbol + '.csv', encoding='gbk', skiprows=1,
                     parse_dates=['candle_begin_time'])
    for hold_period, all_factor_list in all_factor_dict.items():
        if hold_period == 'SPECIAL':
            if 'btc' in all_factor_dict[hold_period]:
                # 能进到这里btc这个变量肯定存在，所以不用管pycharm报错，肯定跑的通
                df = pd.merge(df, btc, on='candle_begin_time', how='left')
        else:
            df_factor = read_coin_factor(root_path, hold_period, all_factor_list, coin_symbol.split('-')[0],
                                         symbol_type)
            if df_factor.empty:
                continue
            df = pd.merge(df, df_factor, on='candle_begin_time', how='left')
    # 至此，所有要画图的数据，都在这个df里了。
    # all_factor_dict里在df的col里找不到的数据，说明这个因子没有进行预运算或者原本K里面没有。在画图的时候skip掉就好了

    # ===按照周期生成数据
    # 截取区间内的币种信息
    symbol_df = pd.merge(benchmark, df, on='candle_begin_time', how='left', sort=True, indicator=True)

    # 获取所有的开仓时间点
    open_times = [pd.to_datetime(time_range.split('--')[0]) for time_range in info_df.loc[i, '选中周期']]
    # 获取所有的平仓时间点
    close_times = [pd.to_datetime(time_range.split('--')[1]) for time_range in info_df.loc[i, '选中周期']]

    # 在数据中加入开仓信息
    symbol_df.loc[symbol_df['candle_begin_time'].isin(open_times), 'open_signal'] = open_signal
    # 在数据中加入平仓信息
    symbol_df.loc[symbol_df['candle_begin_time'].isin(close_times), 'close_signal'] = 0

    # 根据symbol_df生成每笔交易信息
    trade_df = get_trade_info(symbol_df, direction)

    # ===按照日线生成数据
    # 截取区间内的币种信息
    day_df = df[df['candle_begin_time'] >= pd.to_datetime(d_start)]
    day_df = day_df[day_df['candle_begin_time'] <= pd.to_datetime(d_end)]
    # 数据resample时的规则
    rule_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    day_df = day_df.resample(rule='1D', on='candle_begin_time').agg(rule_dict).reset_index()

    # 在日线上标记区间开始时间
    if day_df['candle_begin_time'].dt.date.to_list()[0] < pd.to_datetime(start).date():
        day_df.loc[day_df['candle_begin_time'].dt.date == pd.to_datetime(start).date(), 'signal'] = 'start'
    else:
        day_df.loc[0, 'signal'] = 'start'

    # 在日线上标记区间结束时间
    if day_df['candle_begin_time'].dt.date.to_list()[-1] > pd.to_datetime(end).date():
        day_df.loc[day_df['candle_begin_time'].dt.date == pd.to_datetime(end).date(), 'signal'] = 'end'
    else:
        day_df.loc[day_df.index.max(), 'signal'] = 'end'
    error_list = []
    factor_main_list = []
    for each in add_factor_main_list:
        if 'period' in each.keys():
            if f'{each["因子名称"]}_{each["period"]}' in df.columns:
                factor_main_list.append(each)
            else:
                error_list.append(f'{each["因子名称"]}_{each["period"]}')
        else:
            if each['因子名称'] == 'btc':
                factor_main_list.append(each)
            elif each["因子名称"] in df.columns:
                factor_main_list.append(each)
            else:
                error_list.append(each["因子名称"])

    factor_sub_list = []
    for each_sub in add_factor_sub_list:
        e = []
        for each in each_sub['因子名称']:
            if 'period' in each_sub.keys():
                if f'{each}_{each_sub["period"]}' in df.columns:
                    e.append(each)
                else:
                    error_list.append(f'{each}_{each_sub["period"]}')
            else:
                if each == 'btc':
                    e.append(each)
                elif each in df.columns:
                    e.append(each)
                else:
                    error_list.append(each)
        if len(e) > 0:
            _each_sub = each_sub.copy()
            _each_sub['因子名称'] = e
            factor_sub_list.append(_each_sub)

    # 绘制中性策略的开平仓信息
    draw_hedge_signal_plotly(symbol_df, fig_save_path, f'{direction}_{coin_symbol}_{symbol_type}', res_loc=res_loc,
                             day_df=day_df, trade_df=trade_df, add_factor_main_list=factor_main_list,
                             add_factor_sub_list=factor_sub_list, color_dict=color_dict)
    return error_list


if __name__ == '__main__':
    multiple_process = True
    err_list = []
    # 开始绘制多空持仓的图片
    for direction, info_df in draw_info.items():
        # 判断开仓信号是做多还是做空
        open_signal = 1 if direction == '多头' else -1
        fig_save_path = save_path + f'/{direction}行情图/'
        if not os.path.exists(fig_save_path):
            os.mkdir(fig_save_path)
        # 开始遍历每一行数据画图
        if multiple_process:
            err_list = Parallel(n_jobs)(delayed(each_coin_graph_creator)(i) for i in info_df.index)
        else:
            for i in info_df.index:
                err_list.append(each_coin_graph_creator(i))
    err = list(set(element for sublist in err_list for element in sublist))
    if len(err) > 0:
        print(f'因子 {"、".join(list(set(err)))} 无法载入，不在图中展示')
