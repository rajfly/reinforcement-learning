#!/bin/bash
# rllib train -f config/mountain_car/dqn.yaml
# rllib train -f config/mountain_car/a2c.yaml
# rllib train -f config/mountain_car/ppo.yaml

rllib train -f config/breakout/dqn.yaml
# rllib train -f config/breakout/a2c.yaml
# rllib train -f config/breakout/ppo.yaml