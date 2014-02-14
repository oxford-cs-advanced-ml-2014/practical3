__author__ = 'mdenil'

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

class SimpleObjective(object):
    def __init__(self):
        self.domain = np.transpose(np.array([[-2, 2]]))
        self.ndim = 1

    def map_params(self, x):
        return {'x': float(x)}

    def __call__(self, x):
        return x ** 2 / 100 - x / 100 + np.sin(2 * (x - 0.3))


class Hartmann3Objective(object):
    """
    Hartmann function.

    The global maximum is:
        x* =  (0.114614, 0.555649, 0.852547),
        f(x*) = 3.86278
    """

    def __init__(self):
        self.domain = np.transpose(np.array([[0, 1], [0, 1], [0, 1]]))
        self.ndim = 3

    def map_params(self, x):
        return {'x0': float(x[0]), 'x1': float(x[1]), 'x2': float(x[2])}

    def __call__(self, x):
        x = x.ravel()

        a = np.array([[3., 10., 30.],
                      [0.1, 10., 35.],
                      [3., 10., 30.],
                      [0.1, 10., 35.]])

        c = np.array([1., 1.2, 3., 3.2])

        p = np.zeros((4, 3))

        p[0,0]=0.36890;p[0,1]=0.11700;p[0,2]=0.26730
        p[1,0]=0.46990;p[1,1]=0.43870;p[1,2]=0.74700
        p[2,0]=0.10910;p[2,1]=0.87320;p[2,2]=0.55470
        p[3,0]=0.03815;p[3,1]=0.57430;p[3,2]=0.88280

        s = 0.

        for i in range(4):
            sm = 0.
            for j in range(3):
                sm = sm + a[i,j]*(x[j]-p[i,j])**2.
            s = s + c[i] * np.exp(-sm)

        return s


class RandomForestObjective(object):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        self.ndim = 5

        self.domain = np.transpose(np.array(
            [[1, 10],
             [1, 100],
             [1, 200],
             [1, 100],
             [1, 100]]))

    def map_params(self, params):
        """
        Map continuous parameters to discrete parameters that could be passed into
        RandomForestClassifier.
        """
        params = np.round(params.ravel())

        params_dict = {'n_estimators': int(params[0]),
                       'min_samples_split': int(params[1]),
                       'max_depth': int(params[2]),
                       'min_samples_leaf': int(params[3]),
                       'max_features': int(params[4])}

        return params_dict

    def __call__(self, x):
        random_forest = RandomForestClassifier(
            bootstrap=False,
            **self.map_params(x))

        # 3 fold stratified cross validation
        # http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.cross_val_score.html
        scores = cross_validation.cross_val_score(
            random_forest, self.X_train, self.y_train,
            scoring='accuracy')

        return scores.mean()
