from .dense import Dense
from .conv import Conv
from .flatten import Flatten
from .concatenate import Concatenate


def get_layer(identifier):
    identifier = identifier.lower()
    layers = {
    "dense": Dense,
    "conv": Conv,
    "flatten": Flatten,
    "concatenate": Concatenate
    }

    if identifier not in layers:
        raise ValueError(f"unkown layer: {identifier}")
    return layers[identifier]
