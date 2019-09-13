import pymc3 as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

SUPPLIER_YIELD = [0.9, 0.5, 0.8]
SUPPLIER_YIELD_SD = [0.1, 0.2, 0.2]
PRICES = [220., 100., 120.]
MAX_ORDER_SIZE = [100, 80, 100]
N_OBS = [30, 20, 2]
SALES_PRICE = 500
HOLDING_COST = 100

np.random.seed(1)
data = []
for supplier_yield, supplier_yield_sd, n_obs in zip(SUPPLIER_YIELD, SUPPLIER_YIELD_SD, N_OBS):
    temp = pm.Beta.dist(mu=supplier_yield, sd=supplier_yield_sd, shape=n_obs).random()
    data.append(temp)
data_df = pd.DataFrame(data).T
data_tidy = data_df.unstack().to_frame('yield')
data_tidy.index = data_tidy.index.set_names(['supplier', 'obs'])

@np.vectorize
def loss(in_stock, demand, buy_price, sales_price=SALES_PRICE, holding_cost=HOLDING_COST):
    margin = sales_price - buy_price
    if in_stock > demand:
        total_profit = demand * margin
        total_holding_cost = (in_stock - demand) * holding_cost
        reward = total_profit - total_holding_cost
    else:
        reward = in_stock * margin
    return -reward

def calculate_yield_and_price(orders, supplier_yield=np.array([0.9, 0.5, 0.8]),prices=PRICES):
    orders = np.asarray(orders)
    full_yield = np.sum(orders*supplier_yield)
    price_per_item = np.sum(orders*prices)/np.sum(orders)

    return full_yield, price_per_item


def objective(orders, supplier_yield=supplier_yield_sd_post_pred,
              demand_samples=demand_samples, max_order_size=MAX_ORDER_SIZE):
    orders = np.asarray(orders)
    losses = []
    if np.any(orders<0):
        return np.inf
    if np.any(orders>MAX_ORDER_SIZE):
        return np.inf

    for i, supplier_yield_sample in supplier_yield.iterrows():
        full_yield, price_per_item = calculate_yield_and_price(orders, supplier_yield=supplier_yield_sample)
        loss_i = loss(full_yield, demand_samples[i], price_per_item)
        losses.append(loss_i)
    return np.asarray(losses)

from scipy import optimize

bounds = [(0,max_order) for max_order in MAX_ORDER_SIZE]
starting_value = [50., 50. ,50.]
opt_stoch = optimize.minimize(lambda *args: np.mean(objective(*args)), starting_value=starting_value, bounds=bounds

print(1)