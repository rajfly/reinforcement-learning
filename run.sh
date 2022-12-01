#!/bin/bash
# rllib train -f config/cartpole/impala.yaml
# rllib train -f config/cartpole/apex.yaml
rllib train -f config/cartpole/sac.yaml

# rllib train -f config/cartpole/dqn.yaml
# rllib train -f config/cartpole/a2c.yaml
# rllib train -f config/cartpole/ppo.yaml