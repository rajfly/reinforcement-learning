#!/bin/bash
rllib train -f experiments/pendulum/dqn.yaml
rllib train -f experiments/pendulum/a2c.yaml
rllib train -f experiments/pendulum/a3c.yaml
rllib train -f experiments/pendulum/ppo.yaml