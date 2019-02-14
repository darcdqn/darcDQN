## Installation
##### Dependencies
- [openAI gym](https://github.com/openai/gym).

##### Instructions
Start by creating a virtual environment in a directory of your choice with [virtualenv](https://virtualenv.pypa.io/en/stable/) or [venv](https://docs.python.org/3/library/venv.html) and enter the virtual environment.

#
Example with `venv`

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
cd gym
pip install -e '.[atari,classic_control]'
```

`atari` and `classic_control` are the two environments tested and used by darcDQN from gym so we can install only them.
