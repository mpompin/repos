import numpy as np
import pandas as pd
from PMIMEsig import PMIMEsig
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

SYMBOLS_DICT = {'XG#': (25, 1), '@FV#': (1000, 10), '@TY#C': (1000, 5), 'BL#': (1000, 5),
                'BD#': (1000, 1), 'OAT#': (1000, 1), 'BTP#': (1000, 1), '@US#C': (1000, 1),
                'SW#': (10, 5), "LF#": (10, 5), 'EX#': (10, 5), "@ES#C": (50, 3), "@NQ#C": (20, 3), "@YM#C": (5, 5),
                # 'XAUUSD':(10000, 1),
                'USDJPY':(100, 1000),'EURUSD':(10000, 100000),'GBPJPY':(100, 1000),'XAUUSD':(10000, 100000)}

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


# @numba.jit(nopython=True)
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

def estimate_causal_network(data):

        rM_temp, ecC_temp = PMIMEsig(allM=data.values, Lmax=5, T=1, nnei=5, nsur=100, alpha=0.05)
        rM_temp_df = pd.DataFrame(rM_temp, index=data.columns.tolist(), columns=data.columns.tolist())
        return rM_temp_df, ecC_temp

def aggregate_driver_response_var_lags(conn_matrix, index_instr_dict):

        # aggregate driver --> response variables with the respect lags
        indices_lags_dict = {instr:[] for instr in index_instr_dict.values()}
        for symbol_index, symbol_name in index_instr_dict.items():
            if len(conn_matrix[symbol_index]) == 0:
                continue
            else:
                response_vars, driver_lags = conn_matrix[symbol_index][:,0], conn_matrix[symbol_index][:,1]
                for response_var, driver_lag in zip(response_vars,driver_lags):
                    indices_lags_dict[index_instr_dict[response_var]].append((symbol_name, int(driver_lag)))
        return indices_lags_dict

def train_classifier(X,y):

    from xgboost import XGBClassifier
    from sklearn.linear_model import RidgeClassifierCV, RidgeCV
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC

    test_params = {
        'max_depth':[3, 4, 5],
                   'min_child_weight':[1, 1.5, 2, 5],
        'n_estimators':[1500,2000],
                   'colsample_bytree':[0.6, 0.7, 0.8, 0.9, 1.],
                   'subsample':[0.6, 0.7, 0.8, 0.9, 1.],
        'gamma':[0. ,0.1,0.2,0.3],
        'reg_alpha': [0.8, 1, 1.1, 1.2],
        'reg_lambda':[0, 0.01, 1, 100]
    }

    # classifier = RidgeClassifierCV(alphas=(0.01, 0.1, 1, 10), fit_intercept=True, normalize=False,scoring=None, cv=3)
    # regressor = RidgeCV(alphas=(0.01, 0.1, 1, 10), fit_intercept=True, normalize=False,scoring=None, cv=3)
    classifier = XGBClassifier(max_depth=4, learning_rate=0.01, n_estimators=1000,min_child_weight=1.0,
                               objective='binary:logistic',colsample_bytree=1.0, subsample=1.0,
                               reg_alpha=1.1, reg_lambda=1.)
    # classifier.fit(X=X, y=y)
    # classifier.fit(X=X, y=y)

    grid_search = GridSearchCV(estimator=classifier, param_grid=test_params, scoring='recall',cv=3,)
    grid_search.fit(X=X, y=y)
    #model classifier
    # model = Sequential()
    # model.add(Dense(3, input_dim=X.shape[1],activation='relu', kernel_initializer='normal'))
    # model.add(Dense(3, activation='relu', kernel_initializer='normal'))
    # model.add(Dense(1, activation='sigmoid', kernel_initializer='normal'))
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.fit(scaled_X, y_binary, batch_size=5, nb_epoch=100)

    return classifier

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

    #set fix parameters
    PERIOD = '30T'
    # ROLLING_WINDOW = 1000
    # STEP = 100

    # print(f'{PERIOD} - ROLLING WINDOW = {ROLLING_WINDOW} - STEP = {STEP}')
    #get data and calculate the log returns
    import pickle
    with open(f'd:/Data/bonds_splitted/splits_time.pkl','rb') as file:
        split_times = pickle.load(file)
        file.close()

    #set helper dictionaries
    INDEX_INSTR_DICT = {index:list(SYMBOLS_DICT.keys())[index] for index in np.arange(len(SYMBOLS_DICT.keys()))}
    # signals = {name:pd.Series(index=data_df.index) for name in data_df.columns.tolist()}
    causalities = []
    for start_date, end_date in split_times['XG#']:
        data_df = pd.DataFrame(columns=SYMBOLS_DICT, dtype=np.float32)
        for symbol in SYMBOLS_DICT:
            print(f'parsing {symbol}')
            temp_ = resample_bars(period=PERIOD, symbol=symbol)
            print(f'start {start_date} - end {end_date}')
            temp_ = temp_[np.logical_and(temp_.index >= start_date, temp_.index <= end_date)]
            temp_ = temp_.iloc[200:-200]
            temp_ = temp_['close']
            if temp_.isna().sum() or temp_.empty > 0:
                print(f'passed')
                break
            else:
                data_df[symbol] = temp_
        returns_df = np.log(data_df).diff()
        returns_df = returns_df.dropna(how='all', axis=0)
        returns_df = returns_df.dropna(how='all', axis=1)
        returns_df = returns_df.dropna(how='any', axis=0)
        try:
            rM_temp, ecC_temp = estimate_causal_network(data=returns_df*1e5)
            indices_lags_dict = aggregate_driver_response_var_lags(conn_matrix=ecC_temp, index_instr_dict=INDEX_INSTR_DICT)
            print(f'start {start_date} - end {end_date} \n {indices_lags_dict}')
            data_for_save = dict(start=start_date, end=end_date, causalities=indices_lags_dict)
            causalities.append(data_for_save)
        except:
            continue
    with open(f'd:/bond_causalities{PERIOD}.pkl','wb') as file:
        pickle.dump(causalities, file, pickle.HIGHEST_PROTOCOL)


    # data_df = data_df[data_df.index > '2010']
    # data_df = data_df.dropna(how='all', axis=1)
    # data_df = data_df[[col for col in data_df.columns if col not in ['OAT#','SW#','LF#']]]
    # data_df = data_df.fillna(method='bfill')
    # returns_df = np.log(data_df).diff()
    # returns_df = returns_df.dropna(how='all',axis=0)





    # def create_dataset_for_classifier(temp_returns, response_var, indices_lags_dict):
    #
    #     #create data for classifier D = (X,y)
    #     X = []
    #     y = []
    #     for row_index in np.arange(5, temp_returns.shape[0]):
    #         X_temp = []
    #         y.append(temp_returns[response_var][row_index])
    #         for driver_var, driver_lag in indices_lags_dict:
    #             temp_ = temp_returns[driver_var].iloc[row_index - driver_lag]
    #             X_temp.append(temp_)
    #         X.append(X_temp)
    #     return X, y
    #
    # #estimate causalities for rolling window with fix step
    # i = 0
    # rM = []
    # ecC = []
    # file_to_write = dict()
    # while i < returns_df.shape[0] - (ROLLING_WINDOW + STEP):
    #     temp_returns = returns_df.iloc[i:i+ROLLING_WINDOW]
    #     print(f'checking causalities from {temp_returns.index[0]} to {temp_returns.index[-1]}')
    #     temp_returns *= 1000
    #     rM_temp, ecC_temp = estimate_causal_network(data=temp_returns)
    #     rM.append(rM_temp)
    #     ecC.append(ecC_temp)
    #
    #     indices_lag_dict = aggregate_driver_response_var_lags(conn_matrix=ecC_temp)
    #     indices_lag_dict['start'] = temp_returns.index[0]
    #     indices_lag_dict['end'] = temp_returns.index[-1]
    #     file_to_write[i] = indices_lag_dict
    #     i += STEP



        # for pair in data_df.columns.tolist():
        #     print(f'testing {pair}')
        #     if len(indices_lag_dict[pair]) == 0:
        #         continue
        #     else:
        #         X_train, y_train = create_dataset_for_classifier(temp_returns=temp_returns, response_var = pair, indices_lags_dict=indices_lag_dict[pair])
        #
        #         X_train = np.asarray(X_train)
        #         y_train = np.asarray(y_train)
        #         standardize X
        #         scaler = StandardScaler()
                # scaled_X_train = scaler.fit_transform(X_train)

                # discretize y
                # y_binary = np.where(y_train > 0, 1, -1)
                # y_binary = LabelEncoder().fit_transform(y_binary,)

                # build classifier (logistic regression) from the dataset
                # that causalities were calculated
                # classifier = train_classifier(X=X_train,y=y_binary)
                # test_returns = returns_df.iloc[i + ROLLING_WINDOW:i + ROLLING_WINDOW+100]
                # X_test, y_test = create_dataset_for_classifier(test_returns,response_var = pair, indices_lags_dict=indices_lag_dict[pair] )
                # # X_test_scaled = scaler.transform(X_test)
                # preds = classifier.predict(X_test)
                # preds = np.where(preds>0, 1, -1)
                # preds_df = pd.DataFrame(data=preds, index=test_returns.index[5:])
                # signals[pair].loc[preds_df.index] = preds_df.values.reshape(-1,)
        # i += STEP

    #backtest
    # for pair in data_df.columns:
    #     print(f'backtesting...{pair}')
    #     bars = resample_bars(period=PERIOD, symbol=pair, field_to_use='trade')
    #     labels = signals[pair].replace(np.nan,0)
    #     pip_value = SYMBOLS_DICT[pair][0]
    #     lot = SYMBOLS_DICT[pair][1]
    #     bars, labels = bars.align(labels, join='inner',axis=0)
    #     pnl, trades = backtest(labels=labels, bars=bars,take_profit=100000, stop_loss=100000, pip_value=pip_value,
    #                            exit_signals=1,commission_amount=0, limit_pips=0,lot=lot)
    #     print(f'PNL = {trades.pnl.sum()}')

    print(1)

if __name__ == '__main__':
    main()