class Activation:
    def __call__(self, x):
        raise NotImplementedError
    
    def derive(self, x):
        raise NotImplementedError