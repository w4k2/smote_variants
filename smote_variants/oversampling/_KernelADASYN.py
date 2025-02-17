import numpy as np

from sklearn.decomposition import PCA

from .._NearestNeighborsWithClassifierDissimilarity import NearestNeighborsWithClassifierDissimilarity
from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['KernelADASYN']

class KernelADASYN(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{kernel_adasyn,
                            author={Tang, B. and He, H.},
                            booktitle={2015 IEEE Congress on Evolutionary
                                        Computation (CEC)},
                            title={KernelADASYN: Kernel based adaptive
                                    synthetic data generation for
                                    imbalanced learning},
                            year={2015},
                            volume={},
                            number={},
                            pages={664-671},
                            keywords={learning (artificial intelligence);
                                        pattern classification;
                                        sampling methods;KernelADASYN;
                                        kernel based adaptive synthetic
                                        data generation;imbalanced
                                        learning;standard classification
                                        algorithms;data distribution;
                                        minority class decision rule;
                                        expensive minority class data
                                        misclassification;kernel based
                                        adaptive synthetic over-sampling
                                        approach;imbalanced data
                                        classification problems;kernel
                                        density estimation methods;Kernel;
                                        Estimation;Accuracy;Measurement;
                                        Standards;Training data;Sampling
                                        methods;Imbalanced learning;
                                        adaptive over-sampling;kernel
                                        density estimation;pattern
                                        recognition;medical and
                                        healthcare data learning},
                            doi={10.1109/CEC.2015.7256954},
                            ISSN={1089-778X},
                            month={May}}

    Notes:
        * The method of sampling was not specified, Markov Chain Monte Carlo
            has been implemented.
        * Not prepared for improperly conditioned covariance matrix.
    """

    categories = [OverSampling.cat_density_estimation,
                  OverSampling.cat_extensive,
                  OverSampling.cat_borderline,
                  OverSampling.cat_classifier_distance]

    def __init__(self,
                 proportion=1.0,
                 k=5,
                 *,
                 nn_params={},
                 h=1.0,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            k (int): number of neighbors in the nearest neighbors component
            nn_params (dict): additional parameters for nearest neighbor calculations
                                use {'metric': 'precomputed'} for random forest induced
                                metric {'classifier_params': {...}} to set the parameters
                                of the RandomForestClassifier
            h (float): kernel bandwidth
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(k, 'k', 1)
        self.check_greater(h, 'h', 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.k = k
        self.nn_params = nn_params
        self.h = h
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'k': [5, 7, 9],
                                  'h': [0.01, 0.02, 0.05, 0.1, 0.2,
                                        0.5, 1.0, 2.0, 10.0]}
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

        X_min = X[y == self.min_label]

        # fitting the nearest neighbors model
        nn = NearestNeighborsWithClassifierDissimilarity(n_neighbors=min([len(X_min), self.k+1]), 
                                                        n_jobs=self.n_jobs, 
                                                        **(self.nn_params), 
                                                        X=X, 
                                                        y=y)
        nn.fit(X)
        indices = nn.kneighbors(X_min, return_distance=False)

        # computing majority score
        r = np.array([np.sum(y[indices[i][1:]] == self.maj_label)
                      for i in range(len(X_min))])

        if np.sum(r > 0) < 2:
            message = ("majority score is 0 for all or all but one "
                       "minority samples")
            _logger.info(self.__class__.__name__ + ": " + message)
            return X.copy(), y.copy()

        r = r/np.sum(r)

        # kernel density function
        def p_x(x):
            """
            Returns minority density value at x

            Args:
                x (np.array): feature vector

            Returns:
                float: density value
            """
            result = 1.0/(len(X_min)*self.h)
            result = result*(1.0/(np.sqrt(2*np.pi)*self.h)**len(X[0]))

            exp_term = np.exp(-0.5*np.linalg.norm(x - X_min, axis=1)**2/self.h)
            return result*np.inner(r, exp_term)

        samples = []
        it = 0

        # parameters of the Monte Carlo sampling
        burn_in = 1000
        periods = 50

        # covariance is used to generate a random sample in the neighborhood
        covariance = np.cov(X_min[r > 0], rowvar=False)

        if len(covariance) > 1 and np.linalg.cond(covariance) > 10000:
            message = ("reducing dimensions due to inproperly conditioned"
                       "covariance matrix")
            _logger.info(self.__class__.__name__ + ": " + message)

            if len(X[0]) <= 2:
                _logger.info(self.__class__.__name__ +
                             ": " + "matrix ill-conditioned")
                return X.copy(), y.copy()

            n_components = int(np.rint(len(covariance)/2))

            pca = PCA(n_components=n_components)
            X_trans = pca.fit_transform(X)

            ka = KernelADASYN(proportion=self.proportion,
                              k=self.k,
                              nn_params=self.nn_params,
                              h=self.h,
                              random_state=self.random_state)

            X_samp, y_samp = ka.sample(X_trans, y)
            return pca.inverse_transform(X_samp), y_samp

        # starting Markov-Chain Monte Carlo for sampling
        x_old = X_min[self.random_state.choice(np.where(r > 0)[0])]
        p_old = p_x(x_old)

        # Cholesky decomposition
        L = np.linalg.cholesky(covariance)

        while len(samples) < n_to_sample:
            x_new = x_old + \
                np.dot(self.random_state.normal(size=len(x_old)), L)
            p_new = p_x(x_new)

            alpha = p_new/p_old
            u = self.random_state.random_sample()
            if u < alpha:
                x_old = x_new
                p_old = p_new
            else:
                pass

            it = it + 1
            if it % periods == 0 and it > burn_in:
                samples.append(x_old)

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'k': self.k,
                'nn_params': self.nn_params,
                'h': self.h,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}
