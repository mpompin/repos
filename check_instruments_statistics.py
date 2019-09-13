import numpy as np
import pandas as pd
import hurst as hst
import os

WRITE_PATH = 'd:/'
PERIOD = ['1T','5T', '30T', '1h', '2h', '4h', '12h', '24h']
WINDOW_SIZE = 500
STEP = 100
SYMBOLS = ['EURCAD','EURGBP','GBPCAD','AUDCAD','NZDUSD',
           'USDCAD','AUDCHF','CADCHF','EURUSD','GBPUSD','EURAUD','GBPCHF','USDCHF',
           'BD#','BL#','BTP#','EX#','EZ#','LF#','LG#','XAUUSD','XG#','@FV#',
           '@TY#C','OAT#','@US#C',"@ES#C", "@NQ#C", "@YM#C"]

# SYMBOLS = ['BD#','BL#','BTP#','EX#','EZ#','LF#','LG#',]

def resample_bars(period, symbol, field_to_use='trade'):
    if field_to_use == 'trade':
        what_to_use = ''
    elif field_to_use == 'ask':
        what_to_use = '_ask'
    else:
        what_to_use = '_bid'
    store = pd.HDFStore(f'D:/Data/{symbol}_minutes{what_to_use}.hdf5', mode='r')
    bars = store[symbol]

    store.close()
    if os.path.exists('D:/Data/' + symbol + '_minutes_ask.hdf5'):
        store = pd.HDFStore('D:/Data/' + symbol + '_minutes_ask.hdf5', mode='r')
        bars_ask = store[symbol]
        store.close()
        bars, bars_ask = bars.align(bars_ask, join='inner', axis=0)
        bars['close_ask'] = bars_ask['close']
        bars['open_ask'] = bars_ask['open']
        bars['low_ask'] = bars_ask['low']
    if os.path.exists('D:/Data/' + symbol + '_minutes_bid.hdf5'):
        store = pd.HDFStore('D:/Data/' + symbol + '_minutes_bid.hdf5', mode='r')
        bars_bid = store[symbol]
        store.close()
        bars, bars_bid = bars.align(bars_bid, join='inner', axis=0)
        bars['close_bid'] = bars_bid['close']
        bars['open_bid'] = bars_bid['open']
        bars['high_bid'] = bars_bid['high']

    if period[:-1] == '24':
        base = 17
    elif period[:-1] == '4':
        base = 1
    else:
        base = 0

    if 'close_bid' in bars.columns:
        bars = bars.resample(period, label='right', closed='right', base=base).agg(
            {'open': 'first', 'low': 'min', 'high': 'max', 'close': 'last', 'close_ask': 'first', 'close_bid': 'first',
             'open_ask': 'first', 'open_bid': 'first', 'high_bid': 'max', 'low_ask': 'min'})

    if 'open_bid' in bars.columns:
        bars = bars.resample(period, label='right', closed='right', base=base).agg(
            {'open': 'first', 'low': 'min', 'high': 'max', 'close': 'last', 'open_ask': 'first', 'open_bid': 'first',
             'high_bid': 'max', 'low_ask': 'min'})
    else:
        bars = bars.resample(period, label='right', closed='right', base=base).agg(
            {'open': 'first', 'low': 'min', 'high': 'max', 'close': 'last'})
    bars = bars.dropna()
    # bars.index = bars.index + pd.Timedelta(hours=7)
    if symbol == 'CHFJPY':
        bars.loc[bars.high > 300, 'high'] = 160
    return bars

def calculate_acf_lag1(df, lag=1):
    from statsmodels.tsa.stattools import acf
    acf_lag_calc = acf(x=df, nlags=1, unbiased=True, alpha=0.05)
    acf_lag1 = acf_lag_calc[0][1]
    conf_int = acf_lag_calc[1][1]
    return acf_lag1, conf_int

def calculate_high_low(df):

    high_values = df.high
    low_values = df.low
    return np.median(high_values - low_values)

def calculate_entropy():
    '''
    get entropy with bins:
    potential_profit = (high-low) - spread - commission
    p = prob(potential profit)
    up bar with potential profit > 0
    down bar with potential profit > 0
    else one bin
    '''
    pass

def new_sampler(df):
    t_events = []
    s_higher = 0
    s_lower = 0
    diff = df.diff()
    for tick in diff.index[1:]:
        s_higher = max(0, s_higher + diff.loc[tick])
        s_lower = min(0, s_lower + diff.loc[tick])
        if s_higher > 10e-4:
            t_events.append(tick)
            s_higher = 0
        elif s_lower < 10e-4:
            t_events.append(tick)
            s_lower = 0
    return pd.DatetimeIndex(t_events)

def calculate_true_range_adx(high, low, close):
    true_range = np.zeros(len(close))
    for i in range(1,len(close),1):
        true_range[i] = max(high[i]- low[i],abs(high[i] - close[i-1]),abs(low[i]-close[i-1]))
    return  true_range

def calculate_atr(df, days, **kwargs):
    x = pd.DataFrame(index=df.index)
    x['true_range'] = calculate_true_range_adx(high=df['high'].values, low=df.low.values, close=df.close.values)
    avgtruerange = x['true_range'].rolling(window=days, min_periods=1).mean()
    return avgtruerange

def main():

    for symbol in SYMBOLS:
        acf_lag1_dict = {period: [] for period in PERIOD}
        high_low_dict = {period: [] for period in PERIOD}
        hurst = {period: [] for period in PERIOD}
        mad = {period: [] for period in PERIOD}
        for period in PERIOD:
            print(f'checking {period} - {symbol}')
            data = resample_bars(period=period, symbol=symbol, field_to_use='trade')
            close_values = data.close
            close_log_returns = np.log(close_values).diff()
            if close_log_returns.isna().sum() > 0:
                print(f'filling NaN with bfill - {symbol}')
                close_log_returns = close_log_returns.fillna(method='bfill')
            i = 0
            while i < close_log_returns.shape[0]:
                temp_log = close_log_returns.iloc[i:i+WINDOW_SIZE]
                temp_data = data.iloc[i:i+STEP]
                temp_log *= 10000
                temp_data *= 10000
                acf_lag1, conf_int = calculate_acf_lag1(temp_log, lag=1)
                acf_dict = dict(acf=acf_lag1, conf_95=conf_int, start=temp_data.index[0], end=temp_data.index[-1])
                acf_lag1_dict[period].append(acf_dict)
                high_low = calculate_high_low(temp_data)
                high_low_dict[period].append(high_low)
                mad[period].append(temp_log.mad()/10000)
                print(f'{temp_log.index[0]} - {temp_log.index[-1]} acf={acf_lag1} - high-low={high_low} - mad={temp_log.mad()/10000}')
                if '#' not in symbol:
                    pass
                    entropy = calculate_entropy()
                i += STEP
            acf_lag1_df = pd.DataFrame.from_dict(acf_lag1_dict[period])
            high_low_df = pd.DataFrame.from_dict(high_low_dict[period])
            mad_df = pd.DataFrame.from_dict(mad[period])
            acf_lag1_df.to_csv(f'{WRITE_PATH}{symbol}{period}acf1.csv')
            high_low_df.to_csv(f'{WRITE_PATH}{symbol}{period}highlow.csv')
            mad_df.to_csv(f'{WRITE_PATH}{symbol}{period}mad.csv')


if __name__ == '__main__':
    main()