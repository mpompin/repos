import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import pickle

SYMBOLS_DICT = {'XG#': (25, 1), '@FV#': (1000, 10), '@TY#C': (1000, 5), 'BL#': (1000, 5),
                'BD#': (1000, 1), 'OAT#': (1000, 1), 'BTP#': (1000, 1), '@US#C': (1000, 1),
                'SW#': (10, 5), "LF#": (10, 5), 'EX#': (10, 5), "@ES#C": (50, 3), "@NQ#C": (20, 3), "@YM#C": (5, 5),
                }

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

def main():
    period = '1T'
    # temp = dict()
    # for symbol in SYMBOLS_DICT.keys():
    #     print(symbol)
    #     temp[symbol] = []
    #     data = resample_bars(period=period, symbol=symbol)
    #     start_date = data.index[0]
    #     split_date = start_date
    #     for row in data[data.index >= split_date].index:
    #         if row.month%3 == 0 and split_date.month != row.month:
    #             print(f'start:{split_date} - end_date:{row}')
    #             temp[symbol].append([split_date,row])
    #             split_date = row
    #         else:
    #             continue
    # with open(f'd:/splits_time.pkl', 'wb') as file:
    #     pickle.dump(temp, file, pickle.HIGHEST_PROTOCOL)
    #     print('splits times pickle saved')

    with open(f'd:/splits_time.pkl', 'rb') as file:
        temp = pickle.load(file)
        file.close()
    for symbol in SYMBOLS_DICT.keys():
        print(f'splitting {symbol}')
        data = resample_bars(period=period, symbol=symbol)
        split_dates = temp[symbol]
        for start_date, end_date in split_dates:
            print(f'start:{start_date} - end_date:{end_date}')
            temp_df = data[np.logical_and(data.index >= start_date,data.index <= end_date)]
            temp_df.to_hdf(f'd:/Data/bonds_splitted/{symbol}{start_date.year}{start_date.month}_{end_date.year}{end_date.month}.h5', key=symbol)
print(1)
if __name__ == '__main__':
    main()