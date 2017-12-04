import numpy as np
from components.optimizers import SPSA
from components.models import LinearModel
from components.loss import L2Loss

if __name__ == "__main__":

    # Generate sample points
    N = 1000
    input_dim = 10
    output_dim = 2
    X = np.random.rand(input_dim, N)
    W_true = np.random.rand(output_dim, input_dim)
    print("The true value of W is: \n "+str(W_true))
    Y = np.dot(W_true,X)

    # create the optimizer class
    max_iter = 1000
    optimizer = SPSA(a=9e-1, c=1.0, A=max_iter/10, alpha=0.6, gamma=0.1, loss_function=L2Loss())

    # create the linear model
    linear_model = LinearModel(input_d=input_dim, output_d=output_dim, optimizer=optimizer)

    # the main loop
    for i in range(max_iter):
        linear_model.backward(X,Y)

    # finally print W
    print("The solution is: \n"+str(linear_model.W))