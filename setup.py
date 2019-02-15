from setuptools import setup, find_packages

setup(name='darcDQN',
      version='0.0.1',
      packages=find_packages(),
      install_requires=['gym[atari,classic_control]', 'tensorflow'],
)

