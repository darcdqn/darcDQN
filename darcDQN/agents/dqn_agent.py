from random import randrange

from .abstract_agent import AbstractAgent


class DQNAgent(AbstractAgent):

    """
    Description:
        A deep Q-learning agent, who tries to maximize to total accumulated
        discounted reward over all future time steps given the current state.
    """

    def __init__(self, n_inputs, n_outputs):
        super().__init__(n_inputs, n_outputs)

    def get_action(self, state):
        return randrange(self.n_outputs)


