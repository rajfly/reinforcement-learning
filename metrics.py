import os
import numpy as np
import pandas as pd
from ast import literal_eval
from scipy.stats import iqr, scoreatpercentile

# returns rewards from all episodes for a particular training run
def get_episode_rewards(path):
    df = pd.read_csv(path)
    df['hist_stats/episode_reward'] = df['hist_stats/episode_reward'].apply(lambda x: literal_eval(x))
    return np.hstack(df['hist_stats/episode_reward'].to_numpy())

# returns batches from numpy array
def get_batches(arr, batch_size):
    for idx in range(0, arr.shape[0], batch_size):
        if (idx+batch_size) <= arr.shape[0]:
            yield arr[idx:idx+batch_size]
        else:
            yield arr[idx:]

#  returns normalization value for array
def get_normalize_range(arr):
    arr_95th = np.percentile(arr, 95)
    arr_start = arr[0]
    return arr_95th - arr_start

# returns the inter quartile range (iqr) from an array
# returns mean iqr from sliding window if window_size is specified
def get_iqr(arr, detrend=False, window_size=None):
    norm = get_normalize_range(arr)
    if detrend: arr = np.diff(arr)
    if window_size is None:
        return iqr(arr) / norm
    else:
        batch_iqr = []
        for batch in get_batches(arr, window_size):
            batch_iqr.append(iqr(batch))
        return np.mean(batch_iqr / norm)

# returns the conditional value at risk (cvar) of an array
def get_cvar(arr, alpha=0.05, differences=False, drawdown=False):
    arr = arr / get_normalize_range(arr)
    if differences:
        arr = np.diff(arr)
    elif drawdown:
        peaks = np.maximum.accumulate(arr)
        arr = peaks - arr
    else:
        pass
    risk_val = scoreatpercentile(arr, 100 * alpha)
    cvar = arr[arr <= risk_val].mean()
    return cvar

if __name__ == '__main__':
    for exp in ['a2c', 'apex', 'dqn', 'impala']:
        exp_path = 'cartpole/' + exp
        for (root, dirs, files) in os.walk(exp_path):
            if 'checkpoint' not in root and exp.upper() in root:
                data_path = root + '/progress.csv'
                if '=tf2_' in data_path: framework = 'tf2'
                elif '=tfe_' in data_path: framework = 'tfe'
                elif '=tf_' in data_path: framework = 'tf'
                elif '=torch_' in data_path: framework = 'torch'
                print('--------------- ' + exp + ' ' + framework + ' ---------------')
                episode_rewards = get_episode_rewards(data_path)
                iqr_val = get_iqr(np.copy(episode_rewards), True, 32)
                print('IQR:', iqr_val)
                cvar = get_cvar(np.copy(episode_rewards), 0.05, True, False)
                print('CVAR (diff):', cvar)
                cvar = get_cvar(np.copy(episode_rewards), 0.05, False, True)
                print('CVAR (draw):', cvar)
