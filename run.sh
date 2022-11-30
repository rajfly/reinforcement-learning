#!/bin/bash
rllib train -f config/pong/dqn.yaml
rllib train -f config/pong/a2c.yaml
rllib train -f config/pong/ppo.yaml