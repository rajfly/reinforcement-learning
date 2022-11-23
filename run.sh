#!/bin/bash
rllib train -f experiments/dqn.yaml
rllib train -f experiments/a2c.yaml
rllib train -f experiments/a3c.yaml
rllib train -f experiments/ppo.yaml