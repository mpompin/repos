import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
from sklearn.neighbors import KDTree


WRITE_PATH = 'd:/lap_var_new/'
CUSTOM_TIMEDELTA = 7
PERIOD = '4h'
M = 1
DECISION_THRESHOLD = 0 #threshold to change position
BID_ASK = 1
EXIT_SIGNALS = 0
SYMBOLS = ['EURCAD','EURGBP','GBPCAD','AUDCAD','NZDUSD',
           'USDCAD','AUDCHF','CADCHF','EURUSD','GBPUSD','EURAUD','GBPCHF','USDCHF']
SYMBOLS = ['BD#','BL#','BTP#','EX#','EZ#','LF#','LG#','USDJPY','EURUSD','XAUUSD']
K = 5 #12h-->3,21, 4h-->[71]

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

def backtest(labels, bars, take_profit, stop_loss, pip_value, commission_amount,exit_signals, limit_pips, timed_close_window=-1,do_reverse = 1,reset_take_profit = False,slippage = 0,bid_ask = 0,field_to_use='trade',use_limit =0,lot = 1):

    if bid_ask:
        entry_points, exit_points, entry_price, exit_price, trade_type, closes_limit = create_signals(labels, bars['open'].shift(-1).values, bars['high'].values, bars['low'].values,bars['high_bid'].values, bars['low_ask'].values, bars['close'].values,bars['open_ask'].shift(-1).values,bars['open_bid'].shift(-1).values,take_profit, stop_loss, pip_value, commission_amount,
                                                                                                      exit_signals,
                                                                                                      limit_pips,
                                                                                                      timed_close_window=timed_close_window,use_limit=use_limit)
    else:
        entry_points, exit_points, entry_price, exit_price, trade_type, closes_limit = create_signals(labels, bars[
            'open'].shift(-1).values, bars['high'].values, bars['low'].values, bars['high'].values, bars['low'].values,
                                                                                                      bars[
                                                                                                          'close'].values,
                                                                                                      bars[
                                                                                                          'open'].shift(
                                                                                                          -1).values,
                                                                                                      bars[
                                                                                                          'open'].shift(
                                                                                                          -1).values,
                                                                                                      take_profit,
                                                                                                      stop_loss,
                                                                                                      pip_value,
                                                                                                      commission_amount,
                                                                                                      exit_signals,
                                                                                                      limit_pips,
                                                                                                      timed_close_window=timed_close_window,
                                                                                                      use_limit=0)

    trade_type = [('long' if x else 'short') for x in trade_type]
    entry_points_index = [x + 1 for x in entry_points]
    exit_points_index = [x + 1 for x in exit_points]
    print(labels)
    # exit_points_index[-1] = len(labels) - 1
    entry_points = bars.index[entry_points]
    exit_points = bars.index[exit_points]
    if exit_points_index[-1] >= len(bars):
        exit_points_index[-1] -= 1
    entry_points_index = bars.index[entry_points_index]
    exit_points_index = bars.index[exit_points_index]

    trades = pd.DataFrame(data=dict(entry_points=entry_points_index, exit_points=exit_points_index,
                                    entry_price=entry_price, exit_price=exit_price,
                                    trade_type=trade_type))
    print(len(trades))
    trades['pnl'] = trades['entry_price'] - trades['exit_price']
    # Since we subtract exit_price from entry_price, reverse the long positions' pnl
    trades.loc[trades['trade_type'].isin(['long']), 'pnl'] *= -1
    if trades.shape[0] > 1:
        if trades['exit_points'].iloc[-1] == trades['exit_points'].iloc[-2]:
            trades = trades.iloc[:-1]
    closes = pd.Series(closes_limit, index=bars.index)

    trades['pnl'] = trades['pnl'] * lot
    if commission_amount > 0:
        trades['pnl'] = trades['pnl'] - commission_amount
        trades.loc[trades['exit_price'] != trades['entry_price'].shift(-1), 'pnl'] -= commission_amount
    pnl = (closes.diff()).replace([np.inf, -np.inf], np.nan).fillna(0.)
    if trades.entry_points.shape[0] > 0:
        if np.count_nonzero(pnl.index < trades.entry_points.iloc[0]) > 0:
            pnl[pnl.index < trades.entry_points.iloc[0]] = 0
        # longs = self.trades[self.trades.trade_type == 'long']
        shorts = trades[trades.trade_type == 'short']

        for idx, row in shorts.iterrows():
            pnl[(pnl.index >= row.entry_points) & (pnl.index < row.exit_points)] *= -1
        if bid_ask:
            for i in range(0, len(trades) - 1):
                if trades.loc[i]['exit_points'] != trades.loc[i + 1]['entry_points']:
                    pnl[(pnl.index >= trades.loc[i].exit_points) & (pnl.index < trades.loc[i + 1].entry_points)] *= 0
                    pnl[pnl.index == trades.loc[i].exit_points] -= 2 * commission_amount / lot
                if (trades.loc[i]['exit_points'] == trades.loc[i + 1]['entry_points']) and (
                    trades.loc[i]['exit_price'] != trades.loc[i + 1]['entry_price']):
                    if trades.loc[i + 1]['trade_type'] == 'long':
                        pnl.loc[trades.loc[i]['exit_points']] = bars.loc[trades.loc[i]['exit_points']]['close'] - \
                                                                bars.loc[trades.loc[i]['exit_points']]['open_ask']
                    else:
                        pnl.loc[trades.loc[i]['exit_points']] = -bars.loc[trades.loc[i]['exit_points']]['close'] + \
                                                                bars.loc[trades.loc[i]['exit_points']]['open_bid']
        else:
            for i in range(0, len(trades) - 1):
                if trades.loc[i]['exit_points'] != trades.loc[i + 1]['entry_points']:
                    pnl[(pnl.index >= trades.loc[i].exit_points) & (pnl.index < trades.loc[i + 1].entry_points)] *= 0
                    pnl[pnl.index == trades.loc[i].exit_points] -= 2 * commission_amount / lot
                if (trades.loc[i]['exit_points'] == trades.loc[i + 1]['entry_points']) and (
                    trades.loc[i]['exit_price'] != trades.loc[i + 1]['entry_price']):
                    if trades.loc[i + 1]['trade_type'] == 'long':
                        pnl.loc[trades.loc[i]['exit_points']] = bars.loc[trades.loc[i]['exit_points']]['close'] - \
                                                                bars.loc[trades.loc[i]['exit_points']]['open']
                    else:
                        pnl.loc[trades.loc[i]['exit_points']] = -bars.loc[trades.loc[i]['exit_points']]['close'] + \
                                                                bars.loc[trades.loc[i]['exit_points']]['open']

        if exit_points_index[-1] != pnl.index[-1]:
            pnl[pnl.index >= exit_points_index[-1]] *= 0
            pnl[pnl.index == exit_points_index[-1]] = -commission_amount / lot
        if trades.iloc[-1].exit_points == bars.index[-1]:
            if trades.iloc[-1].trade_type == 'short':
                pnl[pnl.index == bars.index[-1]] = (
                                                   (bars.iloc[-1].open - bars.iloc[-1].close)) - commission_amount / lot
            else:
                pnl[pnl.index == bars.index[-1]] = (
                                                   (bars.iloc[-1].close - bars.iloc[-1].open)) - commission_amount / lot
        df = pd.DataFrame(data=pnl)
        df['labels'] = labels
        df.labels = df.labels.shift()
        pnl[trades.entry_points] -= slippage
        pnl = pnl * lot
        pnl[entry_points] -= commission_amount
        # pnl = pnl.shift().fillna(0)
        if exit_signals:
            pnl[df.labels == 0] = 0
            pnl[df.loc[exit_points_index][df.labels == 0].index] = -commission_amount * 2
            # pnl = pnl.shift(-1).fillna(0)
    else:
        pnl *= 0
    # pnl += 1.
    return pnl, trades

def create_signals(labels, opens, highs, lows, highs_bid, lows_ask, closes, opens_ask, opens_bid, take_profit,
                   stop_loss, pip_value, commission_amount, exit_signals, limit_pips, timed_close_window=-1,
                   do_reverse=1, reset_take_profit=False, slippage=0, use_limit=1):
    take_profit /= pip_value
    stop_loss /= pip_value

    long_active = False
    short_active = False
    entry_points = []
    exit_points = []
    entry_price = []
    exit_price = []
    trade_type = []
    cur_tp = 0.
    cur_sl = 0.
    total_inactive_steps = 0
    closes_limit = opens.copy()
    exit_points.append(-1)

    for i in range(closes.shape[0] - 1):
        exit_price_profit = 0
        if long_active:
            if lows[i] < cur_sl:
                exit_price.append(cur_sl)
                exit_points.append(i)
                closes_limit[i] = cur_sl
                exit_price_profit = cur_sl
                long_active = False
            elif highs[i] > cur_tp:
                exit_price.append(cur_tp)
                exit_points.append(i)
                closes_limit[i] = cur_tp
                exit_price_profit = cur_tp
                long_active = False
            else:
                total_inactive_steps += 1

        elif short_active:
            if highs[i] > cur_sl:
                exit_price.append(cur_sl)
                exit_points.append(i)
                closes_limit[i] = cur_sl
                exit_price_profit = cur_sl
                short_active = False
            elif lows[i] < cur_tp:
                exit_price.append(cur_tp)
                exit_points.append(i)
                closes_limit[i] = cur_tp
                exit_price_profit = cur_tp
                short_active = False
            else:
                total_inactive_steps += 1

        if timed_close_window != -1 and total_inactive_steps > timed_close_window:
            exit_price.append(opens[1])
            exit_points.append(i)
            short_active = False
            long_active = False
            total_inactive_steps = 0

        if labels[i] == 1:

            if not long_active:
                if i + 1 < len(closes):
                    if use_limit:  # Check to see if we are using limits. If yes, then we check the current opens we are using plus the limit to the low_ask, high_bid
                        # If no we are using the open_ask, open_bid
                        if opens[i] - (limit_pips * pip_value) >= lows_ask[i + 1]:
                            limit_price = opens[i] - (limit_pips * pip_value)
                            if short_active and do_reverse:
                                exit_price.append(limit_price)
                                exit_points.append(i)
                                short_active = False

                            if not short_active:
                                entry_points.append(i)
                                # if exit_points[-1] < i:
                                if (closes_limit[i] == cur_tp) or (closes_limit[i] == cur_sl):
                                    1
                                else:
                                    closes_limit[i] = limit_price

                                # if exit_points[-1] == i and (
                                #         (stop_loss < 100) or (take_profit < 100)) and exit_price_profit != 0:
                                #     closes_limit[i] = exit_price_profit
                                #     exit_price_profit = 0

                                entry_price.append(limit_price)
                                trade_type.append(True)
                                long_active = True
                                cur_tp = limit_price + take_profit
                                cur_sl = limit_price - stop_loss
                                total_inactive_steps = 0
                    else:
                        limit_price = opens_ask[i]
                        if short_active and do_reverse:
                            exit_price.append(limit_price)
                            exit_points.append(i)
                            short_active = False

                        if not short_active:
                            entry_points.append(i)
                            if (closes_limit[i] == cur_tp) or (closes_limit[i] == cur_sl):
                                1
                            else:
                                closes_limit[i] = limit_price
                            # if exit_points[-1] == i and (
                            #         (stop_loss < 100) or (take_profit < 100)) and exit_price_profit != 0:
                            #     closes_limit[i] = exit_price_profit
                            #     exit_price_profit = 0

                            entry_price.append(limit_price)
                            trade_type.append(True)
                            long_active = True
                            cur_tp = limit_price + take_profit
                            cur_sl = limit_price - stop_loss
                            total_inactive_steps = 0
            elif reset_take_profit and cur_tp < opens[i] + take_profit:
                if i < len(closes) - 1:
                    cur_tp = opens[i] + take_profit
                else:
                    cur_tp = closes[i] + take_profit
                total_inactive_steps = 0

        if labels[i] == -1:

            if not short_active:
                if i < len(closes):
                    if use_limit:  # Check to see if we are using limits. If yes, then we check the current opens we are using plus the limit to the low_ask, high_bid
                        # If no we are using the open_ask, open_bid
                        if opens[i] + (limit_pips * pip_value) <= highs_bid[i + 1]:
                            limit_price = opens[i] + (limit_pips * pip_value)
                            if long_active and do_reverse:
                                exit_price.append(limit_price)
                                exit_points.append(i)
                                long_active = False

                            if not long_active:
                                entry_points.append(i)
                                # if exit_points[-1] < i:
                                if (closes_limit[i] == cur_tp) or (closes_limit[i] == cur_sl):
                                    1
                                else:
                                    closes_limit[i] = limit_price

                                # if exit_points[-1] == i and (
                                #         (stop_loss < 100) or (take_profit < 100)) and exit_price_profit != 0:
                                #     closes_limit[i] = exit_price_profit
                                #     exit_price_profit = 0
                                entry_price.append(limit_price)
                                trade_type.append(False)
                                short_active = True
                                cur_tp = limit_price - take_profit
                                cur_sl = limit_price + stop_loss
                                total_inactive_steps = 0
                    else:
                        limit_price = opens_bid[i]
                        if long_active and do_reverse:
                            exit_price.append(limit_price)
                            exit_points.append(i)
                            long_active = False

                        if not long_active:
                            entry_points.append(i)
                            # if exit_points[-1] < i:
                            if (closes_limit[i] == cur_tp) or (closes_limit[i] == cur_sl):
                                1
                            else:
                                closes_limit[i] = limit_price

                            # if exit_points[-1] == i and ((stop_loss < 100) or (take_profit < 100)) and exit_price_profit!= 0:
                            #     closes_limit[i] = exit_price_profit
                            #     exit_price_profit = 0
                            entry_price.append(limit_price)
                            trade_type.append(False)
                            short_active = True
                            cur_tp = limit_price - take_profit
                            cur_sl = limit_price + stop_loss
                            total_inactive_steps = 0
            elif reset_take_profit and cur_tp > opens[i + 1] - take_profit:
                if i < len(closes) - 1:
                    cur_tp = opens[i] - take_profit
                else:
                    cur_tp = closes[i] - take_profit
                total_inactive_steps = 0

        if (exit_signals == 1) and (labels[i] == 0):
            if short_active or long_active:
                if short_active:
                    limit_price = opens_ask[i]
                    closes_limit[i] = opens_ask[i]
                if long_active:
                    limit_price = opens_bid[i]
                    closes_limit[i] = opens_bid[i]
                exit_price.append(limit_price)
                exit_points.append(i)
                # trade_type.append(0)
                short_active = False
                long_active = False
                total_inactive_steps = 0
    exit_points.pop(0)
    if len(entry_points) > len(exit_points):
        exit_points.append(closes.shape[0] - 1)
        exit_price.append(closes[-1])



    return entry_points, exit_points, entry_price, exit_price, trade_type, closes_limit

#get data
data = pd.DataFrame()
for pair in SYMBOLS:
    print(f'getting data for {pair}')
    temp_ = resample_bars(period=PERIOD, symbol=pair,field_to_use='trade')
    temp_.columns = [str(col + pair) for col in temp_.columns]
    close = temp_[[col for col in temp_.columns if 'close' in col]]
    data = pd.concat([data,close],axis=1)
data.dropna(axis=0,inplace=True)
data.set_index(data.index + datetime.timedelta(hours=CUSTOM_TIMEDELTA), inplace=True)

#calculate log returns
returns_df = (data).diff().dropna()
# returns_df = np.log(data).diff().dropna()
# trade_columns = [col for col in data.columns if np.logical_or(np.logical_or('close' in col,'high_low' in col),np.logical_or('spread' in col,'ma' in col))]
# trade_data = data[trade_columns]

#get embedded data
embedded_data = returns_df.copy()
for col in embedded_data.columns:
    for m_ in np.arange(1,M):
        embedded_data[col+'lag'+str(m_)] = embedded_data[col].shift(m_)
embedded_data.dropna(inplace=True)
print(f'running LAP_VAR on {SYMBOLS} - M = {M} - K = {K}')

embedded_data['hour'] = embedded_data.index.hour
# embedded_data['weekday'] = embedded_data.index.weekday
embedded_data_scaled = (embedded_data - embedded_data.min(axis=0))/(embedded_data.max(axis=0) - embedded_data.min(axis=0))
# from scipy.spatial.distance import cdist
# row_to_row_distances = cdist(embedded_data_scaled, embedded_data_scaled, metric='chebyshev')

#split data into train/test set
data_length = embedded_data.shape[0]
train_set = embedded_data.iloc[:int(data_length*0.7)]
test_set = embedded_data.iloc[int(data_length*0.7)+1:]
train_set_scaled = embedded_data_scaled.iloc[:int(data_length*0.7)]
test_set_scaled = embedded_data_scaled.iloc[int(data_length*0.7)+1:]

#find neighbors for each row in train set
start_search_point = int(data_length * 0.1)
neighbors = {idx:{'neighbors':None, 'distances':None} for idx in np.arange(start_search_point, train_set.shape[0]-1)}
search_dataset = train_set_scaled.iloc[:start_search_point] #start search in the
for each_row in np.arange(start_search_point, train_set.shape[0]-1):
    print(f'looking neighbors for {train_set_scaled.index[each_row]}')
    current_row = train_set_scaled.iloc[each_row]
    tree = KDTree(search_dataset, leaf_size=1, metric='chebyshev')
    dists, indices = tree.query(current_row.values.reshape(1, -1), k=int(np.sqrt(search_dataset.shape[0])))
    neighbors[each_row]['neighbors'] = indices[0]
    neighbors[each_row]['distances'] = dists[0]
    search_dataset = search_dataset.append(current_row)
    search_dataset = search_dataset.drop_duplicates(keep='last')

#find next bar direction for each symbol and real direction
knn_direction = {idx: {symbol: 0 for symbol in SYMBOLS} for idx in neighbors.keys()}
real_next_bar_direction = {idx: {symbol: 0 for symbol in SYMBOLS} for idx in neighbors.keys()}
for each_row in neighbors.keys():
    local_neighbors = neighbors[each_row]['neighbors'].copy()
    local_neighbors += 1 #check next bar
    next_bars = train_set.iloc[local_neighbors]
    for symbol in SYMBOLS:
        neighbors_next_bar_direction = np.sign(next_bars[f'close{symbol}'])
        knn_direction[each_row][symbol] = np.sum(neighbors_next_bar_direction)
        real_next_bar_direction[each_row][symbol] = np.sign(train_set[f'close{symbol}'].iloc[each_row + 1])
        print(f'Neighbors of {train_set.index[each_row]} for {symbol} moved {knn_direction[each_row][symbol]} - Real = {real_next_bar_direction[each_row][symbol]} ')

SYMBOLS = [symbol for symbol in SYMBOLS if '#' in symbol]
#filter models decision checking its effectiveness
#pass if ma10 of last decision is correct > 59%
filter = {idx: {symbol: 0 for symbol in SYMBOLS} for idx in neighbors.keys()}
for each_row in list(neighbors.keys())[10:]:
    for symbol in SYMBOLS:
        correct_counter = 0
        for past_rows in np.arange(each_row-10, each_row):
            if np.sign(knn_direction[past_rows][symbol]) == real_next_bar_direction[past_rows][symbol]:
                correct_counter += 1
        if correct_counter/np.arange(each_row-10, each_row).shape[0] >=0.6:
            filter[each_row][symbol] = 1

filtered_decisions = {idx: {symbol: 0 for symbol in SYMBOLS} for idx in neighbors.keys()}
for each_row in list(neighbors.keys()):
    for symbol in SYMBOLS:
        filtered_decisions[each_row][symbol] = filter[each_row][symbol] * np.sign(knn_direction[each_row][symbol])

#backtest strategy
for symbol in SYMBOLS:
    labels = []
    for each_row in neighbors.keys():
        labels.append(filtered_decisions[each_row][symbol])
    temp_data = resample_bars(period=PERIOD, symbol=symbol)
    temp_data = temp_data.loc[embedded_data.index[0]:]
    temp_data = temp_data.iloc[:len(labels)]
    backtest(labels=labels, bars=temp_data, bid_ask=0, )


# #check how many times LAP is correct for each symbol
# lap_ppr = {symbol: 0 for symbol in SYMBOLS }
# correct_predictions = {idx: {symbol: 0 for symbol in SYMBOLS} for idx in neighbors.keys()}
# count_row = 0
# for each_row in neighbors.keys():
#     count_row += 1
#     for symbol in SYMBOLS:
#         # correct_predictions[each_row][symbol] = filtered_decisions[each_row][symbol] == real_next_bar_direction[each_row][symbol]
#         correct_predictions[each_row][symbol] = np.sign(knn_direction[each_row][symbol]) == real_next_bar_direction[each_row][symbol]
#         lap_ppr[symbol] += np.sign(knn_direction[each_row][symbol]) == real_next_bar_direction[each_row][symbol]
# # lap_ppr = {k:np.median(v.values())/len(SYMBOLS) for k,v in correct_predictions.items() if v.values() !=0 }
# lap_ppr = {k:v/count_row for k,v in lap_ppr.items()}
# print(f'Positive Predictive Rate\n{lap_ppr}')
# print(f'Average Positive Predictive Rate\n{np.mean(list(lap_ppr.values()))}')


# for symbol in SYMBOLS:
#     test_df = pd.DataFrame([(knn_direction[each_row][symbol],real_next_bar_direction[each_row][symbol]) for each_row in knn_direction.keys()])
#     test_df.columns = ['neighbors_move','real_move']
#     pd.pivot_table(test_df, index=['real_move'], values=['neighbors_move'], aggfunc=np.median)
#     test_tidy = test_df.unstack().to_frame(symbol)
#     test_tidy.index = test_tidy.index.set_names(['row','direction'])
#     test_tidy.reset_index().dropna()
#
# mean_distances = [np.mean(neighbors[each_row]['distances']) for each_row in neighbors.keys()]
# correct_predictions_df = pd.DataFrame.from_dict(correct_predictions,).T
# correct_predictions_df['mean_distance'] = mean_distances
#
# #compare true direction with predicted and create y
# #y=1 if correct else 0
# #X = distances
# X = {}
# y_classifier = {idx: 0 for idx in neighbors.keys()}
# # y_classifier = {idx: {symbol: 0 for symbol in SYMBOLS} for idx in neighbors.keys()}
# for each_row in neighbors.keys():
#     X[each_row] = neighbors[each_row]['distances']
#     count_correct = 0
#     for symbol in SYMBOLS:
#         if np.sign(knn_direction[each_row][symbol]) == real_next_bar_direction[each_row][symbol]:
#             count_correct += 1
#     y_classifier[each_row] = count_correct > round(0.7*len(SYMBOLS),0)
#
# #build classifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
#
# param_grid = {
#                 'n_estimators':[200],
#                 'max_depth':[3],
#                 'min_samples_split':[7],
#
# }
# classifier = RandomForestClassifier(n_estimators=1000, max_depth=3, criterion='entropy', max_features='auto',
#                                     min_samples_split=5,)
# grid_search_cv = GridSearchCV(estimator=classifier, param_grid=param_grid,scoring='roc_auc', cv=5)
# X = np.array(list(X.values()))
# X = (X - X.min(axis=0))/(X.max(axis=0) - X.min(axis=0))
# # X = X - X.mean(axis=0)
# y_ = np.array([y_classifier[each_row] for each_row in neighbors.keys()],dtype=np.float)
# grid_search_cv.fit(X,y=y_)

print(1)

#classify data with X=distances, y = correct prediction
