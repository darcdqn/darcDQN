
import numpy as np


class Berries:
    """A desciption of all possible berries made up from 2**3 different
    tastes and whether or not they are good to eat

    Example berry:
        ([sweet, sour, bitter], 'good to eat')
    """
    def __init__(self):
        self.berries = self._get_berries()
        self.n_berries = len(self.berries)

    def _get_berries(self):
        berries = np.array([
            (np.array([False, False, False]), False),
            (np.array([False, False, True]), False),
            (np.array([False, True, False]), True),
            (np.array([True, False, False]), True),
            (np.array([False, True, True]), False),
            (np.array([True, False, True]), True),
            (np.array([True, True, False]), True),
            (np.array([True, True, True]), True)])
        return berries

    def get_n_berries(self):
        return self.n_berries

    def get_berry(self, index):
        berries = self.berries
        berry = berries[index]
        return berry
