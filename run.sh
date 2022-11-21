#!/bin/bash
rm -rf default
rm nohup.out
rllib train --run DQN --env CartPole-v0 --local-dir "~/projects/reinforcement-learning" --config '{"num_gpus": 1, "framework": {"grid_search": ["tf", "tfe", "tf2", "torch"]}}'