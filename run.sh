#!/bin/bash
rllib train -f config/pendulum/dqn.yaml
rllib train -f config/pendulum/a2c.yaml
rllib train -f config/pendulum/a3c.yaml
rllib train -f config/pendulum/ppo.yaml