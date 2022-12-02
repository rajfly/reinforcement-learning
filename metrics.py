import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import iqr, scoreatpercentile
import matplotlib.pyplot as plt

# returns rewards, timesteps and time for a particular training run
def get_episode_info(path):
    df = pd.read_csv(path)

    df['episode_reward_mean'] = df['episode_reward_mean'].astype(float)
    df['timesteps_total'] = df['timesteps_total'].astype(float)
    df['time_total_s'] = df['time_total_s'].astype(float)

    df = df[df['episode_reward_mean'].notna()]
    df = df[df['timesteps_total'].notna()]
    df = df[df['time_total_s'].notna()]

    episode_reward_mean = df['episode_reward_mean'].to_numpy()
    timesteps_total = df['timesteps_total'].to_numpy()
    time_total_s = df['time_total_s'].to_numpy()
    
    return episode_reward_mean, timesteps_total, time_total_s

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
def get_iqr(rewards, timesteps, detrend=False, window_size=None):
    norm = get_normalize_range(rewards)
    if detrend: 
        rewards = np.diff(rewards)
        timesteps = np.diff(timesteps)
        rewards = np.true_divide(rewards, timesteps)
    if window_size is None:
        return iqr(rewards) / norm
    else:
        batch_iqr = []
        for batch in get_batches(rewards, window_size):
            batch_iqr.append(iqr(batch))
        return np.mean(batch_iqr / norm)

# returns the conditional value at risk (cvar) of an array
def get_cvar(rewards, timesteps, alpha=0.05, differences=False, drawdown=False):
    rewards = rewards / get_normalize_range(rewards)
    if differences:
        rewards = np.diff(rewards)
        timesteps = np.diff(timesteps)
        rewards = np.true_divide(rewards, timesteps)
    elif drawdown:
        peaks = np.maximum.accumulate(rewards)
        rewards = peaks - rewards
    else:
        pass
    risk_val = scoreatpercentile(rewards, 100 * alpha)
    cvar = rewards[rewards <= risk_val].mean()
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

    tf_exp_times = []
    tfe_exp_times = []
    tf2_exp_times = []
    torch_exp_times = []

    for exp in tqdm(['a2c', 'apex', 'dqn', 'impala', 'ppo', 'appo', 'pg', 'ars', 'sac']):
        exp_iqr_val = []
        exp_cvar_diff_val = []
        exp_path = 'cartpole/' + exp

        # get data from training files and calculate iqr, cvar
        for (root, dirs, files) in os.walk(exp_path):
            if 'checkpoint' not in root and exp.upper() in root:
                data_path = root + '/progress.csv'
                episode_rewards, episode_timesteps, episode_times = get_episode_info(data_path)
                iqr_val = get_iqr(np.copy(episode_rewards), np.copy(episode_timesteps), True, 10)
                cvar_diff = get_cvar(np.copy(episode_rewards), np.copy(episode_timesteps), 0.05, True, False)
                exp_total_time = episode_times[-1]

                if '=tf2_' in data_path:
                    framework = 'tf2'
                    tf2_exp_times.append(exp_total_time)
                elif '=tfe_' in data_path:
                    framework = 'tfe'
                    tfe_exp_times.append(exp_total_time)
                elif '=tf_' in data_path:
                    framework = 'tf'
                    tf_exp_times.append(exp_total_time)
                elif '=torch_' in data_path:
                    framework = 'torch'
                    torch_exp_times.append(exp_total_time)
                
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

    # plot average time figure
    avg_time_fig_data = {
        'TF': np.mean(tf_exp_times)/60/60,
        'TFE': np.mean(tfe_exp_times)/60/60,
        'TF2': np.mean(tf2_exp_times)/60/60,
        'TORCH': np.mean(torch_exp_times)/60/60}
    
    plt.bar(
        list(avg_time_fig_data.keys()),
        list(avg_time_fig_data.values()),
        color=('#8e98a3', '#12b5cb', '#e52592', '#f9ab00'))
    
    plt.ylabel('Hours')
    plt.title('Average Time')
    plt.savefig('figures/avg_time.png', format='png', bbox_inches="tight")
    plt.clf()