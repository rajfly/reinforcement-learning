# reinforcement-learning
Testing Reinforcement Learning Algorithms

## User Guide
### Set Up Environment
```bash
conda env create -f environment.yml
conda activate rllib
nohup time ./run.sh &> run.out &
```
## Dev Guide
```bash
conda create -n rllib python=3.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install "ray[rllib]" pygame
conda install tensorflow=2.10.0
conda env export --no-builds > environment.yml
```