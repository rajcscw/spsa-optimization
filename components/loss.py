import numpy as np


class L2Loss(object):
    def __call__(self, W, *args):
        # extract the parameters
        X = args[0][0]
        Y = args[0][1]

        # data loss
        pred = np.dot(W,X)
        squared_loss = np.sum((Y - pred)**2, axis=1).reshape((Y.shape[0],-1))
        average_squared_loss = squared_loss / X.shape[1]
        return average_squared_loss
