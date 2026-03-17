import numpy as np
from data_loader import MNetDataset


def split_types(dataset: MNetDataset):
    types = dataset.types
    np.random.shuffle(types)
    n_types = len(types)
    avg_n_types = n_types // 3
    train_types = types[: avg_n_types]
    val_types = types[avg_n_types: avg_n_types * 2]
    test_types = types[avg_n_types * 2:]
    return train_types, val_types, test_types
