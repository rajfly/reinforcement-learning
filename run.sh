#!/bin/bash
# rllib train -f config/cartpole/impala.yaml
# rllib train -f config/cartpole/apex.yaml
# rllib train -f config/cartpole/dqn.yaml
# rllib train -f config/cartpole/a2c.yaml
# rllib train -f config/cartpole/ppo.yaml
rllib train -f config/cartpole/appo.yaml
rllib train -f config/cartpole/pg.yaml
rllib train -f config/cartpole/ars.yaml
rllib train -f config/cartpole/sac.yaml
