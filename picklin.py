import pickle
import numpy as np
import torch
import time
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


def reshape_image(data):
    img_data = data.reshape(len(data), 3, 32, 32)
    img_data = img_data.transpose(0, 2, 3, 1)
    return img_data


startT = time.time()

dict1 = unpickle('CIFAR10/cifar-10-batches-py/data_batch_1')
dict2 = unpickle('CIFAR10/cifar-10-batches-py/data_batch_2')
dict3 = unpickle('CIFAR10/cifar-10-batches-py/data_batch_3')
dict4 = unpickle('CIFAR10/cifar-10-batches-py/data_batch_4')
dict5 = unpickle('CIFAR10/cifar-10-batches-py/data_batch_5')
dict6 = unpickle('CIFAR10/cifar-10-batches-py/test_batch')

arrTrainData = np.concatenate((reshape_image(dict1['data']), reshape_image(dict2['data']), reshape_image(dict3[
                        'data']), reshape_image(dict4['data']), reshape_image(dict5['data'])), axis=0, dtype=int)
arrTrainLabels = np.concatenate((dict1['labels'], dict2['labels'], dict3['labels'], dict4['labels'], dict5['labels']),
                           axis=0, dtype=int)
print(arrTrainData.shape)
print(arrTrainLabels.shape)
DataTrainTensor = torch.from_numpy(arrTrainData)
LabelTrainTensor = torch.from_numpy(arrTrainData)

arrTestData = np.array(dict6['data'], dtype=int)
arrTestLabels = np.array(dict6['labels'], dtype=int)
print(arrTestData.shape)
print(arrTestLabels.shape)
DataTestTensor = torch.from_numpy(arrTestData)
LabelTestTensor = torch.from_numpy(arrTestLabels)

print(f'Full time: {(time.time()-startT):.2f}sec')


def batch_tensor_to_onehot(tensor, classes):
    result = torch.zeros(tensor.shape[0], classes, *tensor.shape[1:], dtype=torch.int64, device='cpu')
    print(tensor.shape)
    result.scatter_(0, tensor.unsqueeze(1), 1)
    return result


class CustDataset(Dataset):
    def __init__(self, data, labels):
        self.x, self.y = data, labels
        self.x = self.x / 255
        self.y = batch_tensor_to_onehot(self.y.to(torch.int64), 10)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]


train_ds = CustDataset(DataTrainTensor, LabelTrainTensor)
test_ds = CustDataset(DataTestTensor, LabelTestTensor)

# train_dl = DataLoader(train_ds, batch_size=batch_size)
# test_dl = DataLoader(test_ds, batch_size=batch_size)
