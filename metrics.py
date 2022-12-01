import os
import numpy as np
import pandas as pd
from ast import literal_eval
from scipy.stats import iqr, scoreatpercentile
import matplotlib.pyplot as plt

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
    tf_iqr_rankings = []
    tfe_iqr_rankings = []
    tf2_iqr_rankings = []
    torch_iqr_rankings = []

    tf_cvar_diff_rankings = []
    tfe_cvar_diff_rankings = []
    tf2_cvar_diff_rankings = []
    torch_cvar_diff_rankings = []

    for exp in ['a2c', 'apex', 'dqn', 'impala', 'ppo', 'appo', 'pg']:
        exp_iqr_val = []
        exp_cvar_diff_val = []
        exp_path = 'cartpole/' + exp

        # get data from training files and calculate iqr, cvar
        for (root, dirs, files) in os.walk(exp_path):
            if 'checkpoint' not in root and exp.upper() in root:
                data_path = root + '/progress.csv'
                if '=tf2_' in data_path: framework = 'tf2'
                elif '=tfe_' in data_path: framework = 'tfe'
                elif '=tf_' in data_path: framework = 'tf'
                elif '=torch_' in data_path: framework = 'torch'
                episode_rewards = get_episode_rewards(data_path)
                iqr_val = get_iqr(np.copy(episode_rewards), True, 32)
                cvar_diff = get_cvar(np.copy(episode_rewards), 0.05, True, False)
                cvar_draw = get_cvar(np.copy(episode_rewards), 0.05, False, True)
                exp_iqr_val.append((iqr_val, framework))
                exp_cvar_diff_val.append((cvar_diff, framework))
        
        # calculate rankings for iqr and cvar
        exp_iqr_val.sort(key=lambda x:x[0])
        exp_cvar_diff_val.sort(key=lambda x:x[0], reverse=True)
        for idx, x in enumerate(exp_iqr_val):
            if x[1] == 'tf2': tf2_iqr_rankings.append(idx+1)
            elif x[1] == 'tfe': tfe_iqr_rankings.append(idx+1)
            elif x[1] == 'tf': tf_iqr_rankings.append(idx+1)
            elif x[1] == 'torch': torch_iqr_rankings.append(idx+1)

        for idx, x in enumerate(exp_cvar_diff_val):
            if x[1] == 'tf2': tf2_cvar_diff_rankings.append(idx+1)
            elif x[1] == 'tfe': tfe_cvar_diff_rankings.append(idx+1)
            elif x[1] == 'tf': tf_cvar_diff_rankings.append(idx+1)
            elif x[1] == 'torch': torch_cvar_diff_rankings.append(idx+1)

    # plot iqr figure
    iqr_fig_data = {
        'TF': np.mean(tf_iqr_rankings),
        'TFE': np.mean(tfe_iqr_rankings),
        'TF2': np.mean(tf2_iqr_rankings),
        'TORCH': np.mean(torch_iqr_rankings)}
    
    plt.bar(
        list(iqr_fig_data.keys()),
        list(iqr_fig_data.values()),
        color=('#8e98a3', '#12b5cb', '#e52592', '#f9ab00'))
    
    plt.ylabel('Mean Rank')
    plt.title('Dispersion Across Time')
    plt.savefig('figures/iqr.png', format='png', bbox_inches="tight")
    plt.clf()

    # plot cvar (diff) figure
    cvar_diff_fig_data = {
        'TF': np.mean(tf_cvar_diff_rankings),
        'TFE': np.mean(tfe_cvar_diff_rankings),
        'TF2': np.mean(tf2_cvar_diff_rankings),
        'TORCH': np.mean(torch_cvar_diff_rankings)}
    
    plt.bar(
        list(cvar_diff_fig_data.keys()),
        list(cvar_diff_fig_data.values()),
        color=('#8e98a3', '#12b5cb', '#e52592', '#f9ab00'))
    
    plt.ylabel('Mean Rank')
    plt.title('Short Term Risk Across Time')
    plt.savefig('figures/cvar_diff.png', format='png', bbox_inches="tight")
    plt.clf()
