import numpy as np

class MeanSquaredError:
    @staticmethod
    def loss(outputs, labels):
            return np.mean(np.power(outputs - labels, 2), axis=0)
    
    @staticmethod
    def derive(outputs, labels, sample_count):
        return (np.multiply(outputs - labels, 2 / sample_count))