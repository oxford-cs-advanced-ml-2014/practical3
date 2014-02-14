__author__ = 'mdenil'

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
from bo import BO
import matplotlib.pyplot as plt
import time

import black_box

if __name__ == '__main__':
    objective = black_box.SimpleObjective()

    x = np.linspace(-2, 2, 200)
    y = objective(x)

    plt.figure()
    plt.show(block=False)

    # Initialize BO
    bo = BO(objective, noise=1e-1)

    # We need to run a few iterations before we start using the GP
    # or there isn't any data to fit the model.
    bo.optimize(num_iters=bo.burn_in)

    for _ in xrange(50):
        # Try the next point.  bo.optimize doesn't start the optimization over,
        # it runs num_iters _more_ iterations from where it left off.
        bo.optimize(num_iters=1)

        # Get predictions for plotting
        y_hat, y_hat_var = bo.predict(x, predict_variance=True)
        y_hat_upper_bound = y_hat + 1.96 * np.sqrt(y_hat_var)
        y_hat_lower_bound = y_hat - 1.96 * np.sqrt(y_hat_var)
        ei = bo.expected_improvement(bo.grid)

        plt.clf()
        plt.plot(x, y) # true function
        plt.plot(x, y_hat) # estimated mean from the gp
        plt.scatter(bo.X, bo.Y) # points we have measured
        plt.fill( # confidence intervals
            np.concatenate([x, x[::-1]]),
            np.concatenate([y_hat_lower_bound, y_hat_upper_bound[::-1]]),
            alpha=.5, fc='b', ec='None', label='95% confidence interval')
        # EI objective for BO.  We show this as lots of little dots because we
        # are using a fixed grid of candidate points.  We're plotting ei / ei.max()
        # because we don't care about the _value_ of EI, just where its maximum
        # is.
        plt.scatter(bo.grid, ei / ei.max(), color='green', s=1)
        # Put a big red X at the best point found so far
        plt.scatter(bo.best_param, bo.best_value, color='red',
                    s=100, marker='x', linewidths=5)
        plt.draw()

        time.sleep(1)

    print "Optimization finished."
    print "Best point found was f({}) = {}".format(bo.best_param, bo.best_value)

    plt.show()

