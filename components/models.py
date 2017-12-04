import numpy as np


class LinearModel:
    def __init__(self, input_d, output_d, optimizer, reg_factor=0.0):
        self.input_d = input_d
        self.output_d = output_d
        self.reg_factor = reg_factor
        self.optimizer = optimizer

        # initialize the weights
        self.W = np.random.randn(self.output_d, self.input_d) * 0.001

    def forward(self, input):
        return np.dot(self.W, input)

    def backward(self, inputs, targets):
        """
        :param input: inputs
        :param target: targets
        :param selection: selected weight
        :return: updated weight for the selected
        
        Minimizes the squared loss
        
        """

        # Update the weight
        self.W = self.optimizer.step(self.W, inputs, targets)