

"""
Split data into a:
    * training set
    * validation set
    * test set
"""

def split_data(data, train_size=0.6, validation_size=0.2, test_size=0.2):
    if train_size + validation_size + test_size != 1.0:
        raise Exception('Split sizes must equal 1.0')

    # Randomize Data
    data = data.sample(frac=1).reset_index(drop=True)
    total_size = len(data)

    # Create sets
    prev = 0

    train_data = data[prev:int(total_size*train_size)]
    prev += int(total_size*train_size)+1

    valid_data = data[prev:prev+int(total_size*validation_size)]
    prev += int(total_size*validation_size)+1

    test_data = data[prev:prev+int(total_size*test_size)]

    return train_data, valid_data, test_data
