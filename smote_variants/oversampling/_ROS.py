import numpy as np

from sklearn.metrics import pairwise_distances

from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['ROS']

class ROS(OverSampling):
    categories = [OverSampling.cat_extensive]

    def __init__(self,
                 proportion=1.0,
                 *,
                 random_state=None):
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0)

        self.proportion = proportion
        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        target_class_indices = np.flatnonzero(y == self.min_label)
        new_X_indices = self.random_state.choice(target_class_indices, size=n_to_sample)

        return (np.vstack([X, X[new_X_indices]]),
                np.hstack([y, np.repeat(self.min_label, len(new_X_indices))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'random_state': self._random_state_init}
