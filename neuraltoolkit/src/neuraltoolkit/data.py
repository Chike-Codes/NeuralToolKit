import numpy as np

def shuffle_data(x, y, seed=None):
    rng = np.random.default_rng(seed)
    rng.shuffle(x)
    rng = np.random.default_rng(seed)
    rng.shuffle(y)
    return x, y


def create_batches(arr:list, batch_size:int):
    count = len(arr) // batch_size
    if count == 0:
        count = 1
    return np.array_split(arr, count)

def split_validation_data(validation_data, validation_split, validation_batch_size):
    if validation_split > 0:
        if validation_split > 1:
            raise ValueError("validation_data must be a fraction between 0 and 1")
        element_count = int(sample_count * validation_split)
        selection = slice(sample_count - element_count, sample_count)
        sample_count -= element_count
        validation_data = (create_batches(x[selection], validation_batch_size), 
                            create_batches(y[selection], validation_batch_size))
        x = np.delete(x, selection, axis=0)
        y = np.delete(y, selection, axis=0)
    elif validation_data != None:
        validation_data = (create_batches(validation_data[0], validation_batch_size), 
                            create_batches(validation_data[1], validation_batch_size))
    return validation_data