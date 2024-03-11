def signal(*args):
    df = args[0]
    n = args[1]
    factor_name = args[2]

    df[factor_name] = df['close'].pct_change().rolling(n, min_periods=1).std()

    return df
