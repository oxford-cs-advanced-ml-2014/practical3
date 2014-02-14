import numpy as np
from sklearn.gaussian_process import GaussianProcess
from scipy import stats
from black_box import sobol


class BO:
    """
    Implements Bayesian optimization with Expected Improvement (EI).
    """

    def __init__(self, objective, noise, burn_in=2, grid=None):
        """
        Initialize the model. Here dim is the dimensionality of the problem.
        """

        self.objective = objective
        self.burn_in = burn_in # check this many random points before fitting the GP

        self.gp = GaussianProcess(theta0=.5, thetaL=.01, thetaU=10., nugget=noise)
        self.X = np.zeros((0, objective.ndim))
        self.Y = np.zeros((0, 1))

        if grid:
            self.grid = grid

        else:
            # If the user did not pass a grid, we construct a grid by using a
            # sobol sequence.

            self.grid = np.transpose(sobol.i4_sobol_generate(
                objective.ndim, objective.ndim * 200, 7))

            # expand the grid to cover the domain of the objective
            d = self.objective.domain
            self.grid = self.grid * (d[1] - d[0]) + d[0]

    @property
    def ndim(self):
        """
        The dimensionality of the objective.
        """
        return self.objective.ndim

    @property
    def n(self):
        """
        The number of times the objective function has been evaluated.
        """
        return self.X.shape[0]

    @property
    def best_param(self):
        """
        The best parameter value found so far.
        """
        return self.X[np.argmax(self.Y)]

    @property
    def best_value(self):
        """
        The value of the best point tried so far.
        """
        return self.Y.max()

    def _ensure_shape(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        # Handle cases where X is a vector
        if X.ndim == 1:
            if self.ndim == 1:
                # If this is a 1d BO then a vector is a list of points to evaluate
                X = X.reshape((-1, 1))
            else:
                # If this is a >1d GP then a vector is a single point
                X = X.reshape((1, -1))

        return X

    def add_points(self, X, Y):
        """
        Add data points to update the model.

        X can be a single vector of size (d,) (or of size (n,) if ndim == 1) or a matrix of size (n,d)
        to evaluate n points at once.

        There should always be exactly one Y value for each X.
        """

        self.X = np.vstack([self.X, self._ensure_shape(X)])
        self.Y = np.vstack([self.Y, Y])

        if self.n >= self.burn_in:
            # Fit the gp after we have self.burn_in observations.
            self.gp.fit(self.X, self.Y)

    def predict(self, X, predict_variance=False):
        """
        Evaluate predictions for the point(s) X.

        X can be a single vector of size (d,) or a matrix of size (n,d) to evaluate n points at once.

        Returns the mean prediction for each point in X.  If predict_variance is True then it will also return the
        variance for each prediction.
        """
        X = self._ensure_shape(X)

        if predict_variance:
            return np.zeros(X.shape[0]), 0.1+np.zeros(X.shape[0])
        else:
            return np.zeros(X.shape[0])

    def expected_improvement(self, X):
        """
        Evaluate expected improvement at the points X.  This corresponds to an
        index strategy of:

            i(x) = E[f(x) - f(x_plus)]

        where 'x_plus' is the best x seen so far.

        X is a (n,d) matrix of d-dimensional points (d == self.ndim).

        Returns a vector of size (n,), where element i is the result of evaluating
        the EI of the point X[i].

        Note:
        You can evaluate the standard normal PDF and CDF using stats.norm.pdf()
        and stats.norm.cdf() respectively.
        """
        X = self._ensure_shape(X)

        return np.random.normal(size=X.shape[0])


    def optimize(self, num_iters, verbose=True):
        """
        Optimize the objective function 'num_iters' iterations by using EI.

        You can call this function again run more iterations of optimization.

        Returns the best point found so far.
        """

        for i in xrange(self.n, self.n + num_iters):
            x_next = self._get_next()
            self.add_points(x_next, self.objective(x_next))

            if verbose:
                print "Iteration {} with best value: {}".format(i, self.best_value)

        return self.best_param

    def _get_next(self):
        """
        Returns the next point to evaluate
        """

        if self.n < self.burn_in:
            # We don't bother with EI until we have at least burn_in observations.
            # Instead of optimizing EI, we pick random points off the grid to
            # evaluate.

            index_opt = self.n + 1
            x_opt = self.grid[index_opt]

        else:
            # Compute EI for all the points on the grid and return the maximizer.
            ei = self.expected_improvement(self.grid)

            index_opt = np.argmax(ei)
            x_opt = self.grid[index_opt]

        # Remove the point from the grid so we don't look there again.
        self.grid = np.delete(self.grid, index_opt, 0)

        return self._ensure_shape(x_opt)