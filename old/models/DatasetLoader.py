import random
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import one_hot

import csv


def make_data_set(file_path):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    labels = []

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            data = [float(l) for l in line[:-1]]
            label = line[-1]

            if label not in labels:
                labels.append(label)

            if random.randint(0, 9) > 6:
                train_data.append(data)
                train_labels.append(labels.index(label))
            else:
                test_data.append(data)
                test_labels.append(labels.index(label))
    
    print('convert data to tensor')
    tensor_train_x = torch.Tensor(train_data)
    tensor_train_y = one_hot(torch.tensor(train_labels), num_classes=len(labels))
    train_set = TensorDataset(tensor_train_x, tensor_train_y)
    train_loader = DataLoader(train_set, batch_size=10, shuffle=True, num_workers=0)

    tensor_test_x = torch.Tensor(test_data)
    tensor_test_y = one_hot(torch.tensor(test_labels), num_classes=len(labels))
    test_set = TensorDataset(tensor_test_x, tensor_test_y)
    test_loader = DataLoader(test_set, batch_size=10, shuffle=True, num_workers=0)

    return train_loader, test_loader, labels