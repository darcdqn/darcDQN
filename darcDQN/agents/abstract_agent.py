from abc import ABC, abstractmethod


class AbstractAgent(ABC):

    """
    Description:
        An agent is a, possibly but not necessarily trainable, policy
        for which action to take given the current state of the system.
    """

    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

    @abstractmethod
    def get_action(self, state):
        """Return which action to take given the current state."""
        pass

    def observe(self, prev_state, next_state, action, reward, done):
        """Observe the results of an action."""
        pass


