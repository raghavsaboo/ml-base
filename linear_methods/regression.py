"""Provides Linear, Logistic, Shrinkage and Subset selection methods.

Includes closed form, Gram-Schmidt procedure, QR decomposition, Multiple-Output
regression etc.
"""

import numpy as np

class BaseRegression:
    """Base class for all regressions"""

    def __init__(self):
        self._trained = False
        self._weights = np.NaN

    @property
    def trained(self):
        return self._trained

    @property
    def weights(self):
        return self._weights

    @trained.setter
    def trained(self, value):
        self._trained = value
    
    @weights.setter
    def weights(self, value):
        self._weights = value

class LinearRegression(BaseRegression):
    """Linear Regression implementation with Least Squares, Gradient Descent,
    Gram-Schmidt procedure, and QR decomposition.
    """

    def predict(self, X, bias=True):
        """
        
        Arguments:
            X {np.ndarray} -- input data of shape [n_samples, n_features]
            bias {boolean} -- whether to include a bias term or not

        Returns:
            np.ndarray: shape [n_samples, 1] predicted values

        Raises:
            ValueError if model has not been fit
        """
        if not self.trained:
            raise NameError('Model has not been trained.')

        X = np.asarray(X)

        # Add column of 1s to X for bias/intercept term.
        if bias: 
            X = np.column_stack((np.ones(np.shape(X)[0]), X))

        prediction = np.dot(X, np.transpose(self.weights))
        return prediction

    def train(self, X, y, bias=True, solver_type='least_squares', reg_param=0):
        """Learn the weights of the linear regression model using the solver
        type selected along with the type of regularization.
        
        Arguments:
            X {np.ndarray} -- input data of shape [n_samples, n_features]
            y {np.ndarray} -- output data of shape [n_samples, 1] 
        
        Keyword Arguments:
            bias {bool} -- include bias/intercept term (default: {True})
            solver_type {str} -- use solver between 
                ('least_squares', 'gradient_descent') (default: {'least_squares'})
            reg_param {int} -- regularization penalty (0, 1, or 2) for no 
                regularization, L1 regularization, and L2 regularization 
                respectively (default: {0})
        
        Returns:
            [type] -- [description]
        """

        y = np.asarray(y)
        X = np.asarray(X)

        if bias:
            X = np.column_stack((np.ones(np.shape(X)[0]), X))

        if solver_type == 'least_squares':
            Xt = np.transpose(X)
            XtX = np.dot(Xt, X)
            regularized = XtX + reg_param * np.identity(len(XtX))
            self.weights = np.dot(np.linalg.pinv(regularized), np.dot(Xt, y))

        self.trained = True
        return self