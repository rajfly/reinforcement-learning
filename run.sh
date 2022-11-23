#!/bin/bash
rm -rf default

# for env in CartPole-v1; do
#     for algo in DQN; do
#         rllib train --run $algo --env $env --checkpoint-at-end --stop '{"timesteps_total": 300000000}' --local-dir "~/projects/reinforcement-learning" --config '{"num_gpus": 1, "num_workers": 15, "num_envs_per_worker": 200, "framework": {"grid_search": ["tf", "tfe", "tf2", "torch"]}}'
#     done
# done

# https://paperswithcode.com/sota/atari-games-on-atari-2600-montezumas-revenge
for env in ALE/MontezumaRevenge-v5; do
    for algo in APEX; do
        rllib train --run $algo --env $env --checkpoint-at-end --stop '{"timesteps_total": 300000000}' --local-dir "~/projects/reinforcement-learning" --config '{"num_gpus": 1, "num_workers": 15, "num_envs_per_worker": 200, "framework": {"grid_search": ["tf", "tfe", "tf2", "torch"]}}'
    done
done