__author__ = 'mdenil'

import numpy as np
from bo import BO
from pprint import pprint
import black_box
import mnist

if __name__ == '__main__':
    objective = black_box.RandomForestObjective(
        X_train=np.load("data/mnistFeatures.npy"),
        y_train=np.load("data/mnistlabels.npy"))

    # Write code to optimize the objective using BO
    bo = ...

    print "Optimization finished."
    print "Best parameter settings found:"
    # objective.map_params(...) will attach names to the parameters so you can tell what they are
    pprint(objective.map_params(bo.best_param))
    print "With cross validation accuracy: {}".format(bo.best_value)

    mnist.visualize_predictions(objective.map_params(bo.best_param))