<div align="center">

[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/kscalelabs/onshape/blob/main/LICENSE)
[![Discord](https://img.shields.io/discord/1224056091017478166)](https://discord.gg/k5mSvCkYQh)
[![Wiki](https://img.shields.io/badge/wiki-humanoids-black)](https://humanoids.wiki)
<br />
[![python](https://img.shields.io/badge/-Python_3.8-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![ruff](https://img.shields.io/badge/Linter-Ruff-red.svg?labelColor=gray)](https://github.com/charliermarsh/ruff)
<br />
</div>

![Zbot Img 1](zbot_img.png)

# K-Scale Genesis Playground Library

This repository implements training for robotic control policies in Genesis simulator. For more information, see the [documentation]([https://docs.kscale.dev/simulation/genesis](https://docs.kscale.dev/docs/genesis#/)).

# ZBot Training and Evaluation Commands

## Commands to run training and evaluation
```sh

# Training
python genesis_playground/zbot/zbot_train.py \
    -e experiment-name \
    --num_envs 1024 \
    --wandb_project project-name \
    --use_wandb True

# Evaluation
python genesis_playground/zbot/zbot_eval.py \
    -e experiment-name \
    --ckpt 300 \
    --show_viewer

```


![Zbot Img 2](zbot_joint_id.png)


