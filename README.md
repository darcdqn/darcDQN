# darcDQN 0.1

This project contains the source code of darcDQN, a Python-based research project for a dynamic architecture to be used with deep reinforcement learning, necessary to produce the results in our M.Sc. thesis paper "Dynamic Architectures with Deep Q-Learning Networks".

To replicate the experiment results, [OpenAI Gym](https://gym.openai.com/) with the [Classic control](https://github.com/openai/gym#classic-control) and [Atari](https://github.com/openai/gym#atari) environments need to be installed. An install script for these dependencies through [`pip`](https://pypi.org/project/pip/) is provided.

## Installation
##### Dependencies
- [OpenAI Gym](https://github.com/openai/gym).

##### Instructions
Start by creating a virtual environment in a directory of your choice with [virtualenv](https://virtualenv.pypa.io/en/stable/) or [venv](https://docs.python.org/3/library/venv.html) and enter the virtual environment.

#
Example with [`venv`](https://docs.python.org/3/library/venv.html)

```
python -m venv venv
```

If you want to use packages already installed on your system, append `--system-site-packages`.

Then, enter your virtual environment with

```
source venv/bin/activate
```
#

Clone [gym](https://github.com/openai/gym) to your directory and install with

```
git clone https://github.com/openai/gym.git
cd gym
pip install -e '.[atari,classic_control]'
```

`atari` and `classic_control` are the two environments tested and used by darcDQN from gym so we only need to install them.
