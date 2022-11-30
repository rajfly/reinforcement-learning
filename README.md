# reinforcement-learning
Testing Reinforcement Learning Algorithms

## User Guide
### Set Up Environment
```bash
conda env create -f environment.yml
conda activate rllib
```
### Start Training
```bash
nohup rllib train -f config/cartpole/a2c.yaml &
nvidia-smi | grep 'ray' | awk '{print $5}' | xargs -n1 kill -9
```
## Dev Guide
```bash
conda create -n rllib python=3.8
pip install "ray[rllib]" pygame "gym[atari]" "gym[accept-rom-license]" atari_py

# for cpu
pip install tensorflow torch

# for gpu
# see tensorflow and pytorch websites and install based on your cuda version
# selab bravo server (4 x A4000)
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install tensorflow=2.10.0
# selab alfa server (4 x 2070 Super with cuda 10.1)
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
conda install tensorflow-gpu=2.2.0

conda env export --no-builds > environment.yml
```