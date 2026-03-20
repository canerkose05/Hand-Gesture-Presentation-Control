from abc import ABC, abstractmethod
import numpy as np


class CostFunction(ABC):
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def cost_function(self, pred, gt):
        pass

    def cost_function_deriv(self, pred, gt, X):
        num_examples = pred.shape[1]
        dZ = pred - gt
        dW = (1 / num_examples) * np.dot(dZ, X.T)
        db = (1 / num_examples) * np.sum(dZ, axis=1, keepdims=True)
        return dW, db


class MSE(CostFunction):
    def __str__(self):
        return "Cost Function: MSE"

    def cost_function(self, pred, gt):
        num_examples = pred.shape[1]
        return (1 / num_examples) * np.sum(np.square(gt - pred))


class CrossEntropy(CostFunction):
    def __init__(self, binary=False):
        self.binary = binary

    def __str__(self):
        mode = "binary" if self.binary else "categorical"
        return f"Cost Function: Cross Entropy - {mode}"

    def cost_function(self, pred, gt):
        num_examples = pred.shape[1]
        pred = np.clip(pred, 1e-15, 1 - 1e-15)

        if self.binary:
            return (-1 / num_examples) * np.sum(
                gt * np.log(pred) + (1 - gt) * np.log(1 - pred)
            )

        return (-1 / num_examples) * np.sum(gt * np.log(pred))