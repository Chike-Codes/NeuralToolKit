class Optimizer:
    def build(self, gradient_shape:tuple):
        raise NotImplementedError
    
    def init(self):
        raise NotImplementedError
    
    def optimize(self, gradient):
        raise NotImplementedError