#!/bin/bash
rllib train -f config/cartpole/impala.yaml
rllib train -f config/cartpole/apex.yaml
rllib train -f config/cartpole/sac.yaml