#!/bin/bash
rm -rf default
# for env in CartPole-v1 Taxi-v3; do
for env in CartPole-v1; do
    # A2C A3C APPO ARS BC CQL DDPG APEX-DDPG ES DQN Rainbow APEX-DQN IMPALA MAML MARWIL PG PPO R2D2 SAC SlateQ TD3
    for algo in A2C A3C; do
        # echo "Environment: $env Algorithm: $algo"
        time rllib train --run $algo --env $env --local-dir "~/projects/reinforcement-learning" --config '{"num_gpus": 1, "framework": {"grid_search": ["tf", "tfe", "tf2", "torch"]}}' &
    done
done