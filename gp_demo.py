# Author: Vincent Dubourg <vincent.dubourg@gmail.com>
#         Jake Vanderplas <vanderplas@astro.washington.edu>
# Modified for teaching purposes: Nando de Freitas, Misha Denil
# Licence: BSD 3 clause

import numpy as np
from sklearn.gaussian_process import GaussianProcess
import matplotlib.pyplot as plt


def objective(x):
    """The function to predict."""
    return x + np.sin(x)

#----------------------------------------------------------------------

# Training input data
X = np.asarray([1., 4., 5., 6., 8., 9.]).reshape((-1, 1))

# Training labels and noise (training output data)
noise_level = 1e-2
y = objective(X) + np.random.normal(0, noise_level)

# Instantiate a Gaussian Process model
# squared exponential is exp(-theta0*distanceSquared)
gp = GaussianProcess(
    corr='squared_exponential',
    theta0=1e0,
    # thetaL=1e-1,
    # thetaU=1e2,
    nugget=noise_level ** 2,
    random_start=10)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction on the meshed x-axis (ask for variance predictions as well)
x = np.linspace(0, 10, 1000).reshape((-1, 1))
y_pred, variance = gp.predict(x, eval_MSE=True)
stddev = np.sqrt(variance)

# Plot the function, the prediction and the 95% confidence interval based on
# the variance
fig = plt.figure()
plt.plot(x, objective(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
plt.errorbar(X.ravel(), y.ravel(), noise_level, fmt='r.', markersize=10, label=u'Observations')
plt.plot(x, y_pred, 'b-', label=u'Mean Prediction')

y_conf_upper_bound = y_pred + 1.96 * stddev
y_conf_lower_bound = y_pred - 1.96 * stddev
plt.fill(
    np.concatenate([x, x[::-1]]),
    np.concatenate([y_conf_upper_bound, y_conf_lower_bound[::-1]]),
    alpha=.5, fc='b', ec='None', label='95% confidence interval')

plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')

plt.show()