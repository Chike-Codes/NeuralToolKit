import numpy as np

class CategoricalCrossEntropy:
    @staticmethod
    def loss(outputs, labels):
           return np.mean((np.log(outputs + 1e-8) * labels * -1), axis=0)
    
    @staticmethod
    def derive(outputs, labels, sample_count):
           return np.multiply(outputs -labels, 1 / sample_count)