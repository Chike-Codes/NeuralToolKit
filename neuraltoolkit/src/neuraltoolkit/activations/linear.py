from .activation import Activation
class Linear(Activation):
    def __call__(self, x):
        return x
    
    def derive(self, x):
        return 1