# reinforcement-learning
Testing Reinforcement Learning Algorithms

## User Guide

### Set Up Environment
```bash
conda create -n rllib python=3.8
conda activate rllib
pip install "ray[rllib]" pygame "gym[atari]" "gym[accept-rom-license]" atari_py

# for cpu
pip install tensorflow torch

# for gpu
# see tensorflow and pytorch websites and install based on your cuda version
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install tensorflow=2.10.0
```

### Configure Training
```yaml
# required fields to modify: local_dir, num_gpus, num_workers
# local_dir: modify '~/projects/' to point to the directory where this repo is located
# num_gpus: number of gpus to allocate to each framework
#           set to 0 for cpu
#           total gpus used is 4 in the example below
# num_workers: number additional workers to allocate to each framework 
#              the framework already has one worker by default
#              each worker uses 1 cpu by default
#              with 5 additional workers, total cpus used is (5 + 1) * 4 the example below
a2c:
  env: CartPole-v1
  run: A2C
  stop:
    timesteps_total: 10000000
  checkpoint_config:
    checkpoint_at_end: true
  local_dir: ~/projects/reinforcement-learning/cartpole
  config:
    num_gpus: 1
    num_workers: 5
    num_envs_per_worker: 5
    eager_tracing: true
    framework:
      grid_search:
        - tf
        - tfe
        - tf2
        - torch
```

### Start Training
```bash
# see nohup.out for train output
nohup rllib train -f config/cartpole/a2c.yaml &
```

### View Training Graphs
```bash
tensorboard --logdir=cartpole 
```

## Dev Utility Commands
```bash
# create and export conda env
conda env create -f environment.yml
conda env export --no-builds > environment.yml

# kill all gpu processes
nvidia-smi | grep 'ray' | awk '{print $5}' | xargs -n1 kill -9
```