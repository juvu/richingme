def signal(*args):
    df = args[0]
    n = args[1]
    factor_name = args[2]

    df['NetBuy'] = df['taker_buy_quote_asset_volume'] * 2 - df['quote_volume']
    df[factor_name] = df['NetBuy'].rolling(n, min_periods=2).std()

    return df
