import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
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


def batch_tensor_to_onehot(tensor, data_classes):
    result = torch.zeros(tensor.shape[0], data_classes, *tensor.shape[1:], dtype=torch.int64, device=device)
    result = result.scatter_(1, tensor.unsqueeze(1), 1).float()
    return result


def reshape_image(data):
    img_data = data.reshape(len(data), 3, 32, 32)
    # img_data = img_data.transpose(0, 2, 3, 1)
    return img_data


class CustDataset(Dataset):
    def __init__(self, data, labels):
        self.x, self.y = data, labels
        self.x = self.x / 255
        self.y = batch_tensor_to_onehot(self.y.to(torch.int64), 10)
        # print(self.y.shape)

    def __len__(self):
        # print(f"CustDataset X: {self.x.shape} \nY:{self.y.shape}")
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class ConvNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)  # MAX pool
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fullcon1 = nn.Linear(16 * 5 * 5, 120)
        self.fullcon2 = nn.Linear(120, 84)
        self.fullcon3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape)
        # print(x)
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        # print(x)
        x = x.view(-1, 16 * 5 * 5)
        # print(x.shape)
        # print(x)
        x = F.relu(self.fullcon1(x))
        # print(x.shape)
        # print(x)
        x = F.relu(self.fullcon2(x))
        # print(x.shape)
        # print(x)
        x = self.fullcon3(x)
        # print(x.shape)
        # print(x)
        return x


def accuracy_test(ds):
    fc = 0
    counter = 0
    for x, y in ds:
        # print(x.shape)
        # print(y.shape)
        f_index = torch.argmax(model(x))
        y_index = torch.argmax(y)
        # print(f_index)
        # print(f"{y_index}\n----")
        if y_index == f_index:
            fc += 1
        counter += 1
    accuracy = (fc / counter)
    print(f'The network had accuracy: {accuracy:.1f}%')


def train_model(dl, model):
    L = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        t_lap = time.time()
        for i, (x, y) in enumerate(dl):
            # input = batch_size images, 3 color values, 32 x 32 pixels
            opt.zero_grad()
            # Onward, pass it forward!
            outputs = model(x)
            loss = L(outputs, y)

            # Optimize
            loss.backward()
            opt.step()

        epoch_list.append(epoch + 1)
        losses.append(loss.item())
        print(f'Epoch: {epoch + 1} | Loss: {loss.item():.3f} | Time: {(time.time() - t_lap):.1f} sec')
    print("Accuracy test with training data: ")
    accuracy_test(train_ds)
    print("Accuracy test with test data: ")
    accuracy_test(test_ds)
    return np.array(epoch_list), np.array(losses)


t_start = time.time()

# Declaring device as GPU
device = torch.device("cpu")
# if torch.cuda.is_available() else "cpu"

# Hyper parameters
epochs = 5
batch_size = 5
learning_rate = 0.01

# All data batches
dict1 = unpickle('CIFAR10/cifar-10-batches-py/data_batch_1')
dict2 = unpickle('CIFAR10/cifar-10-batches-py/data_batch_2')
dict3 = unpickle('CIFAR10/cifar-10-batches-py/data_batch_3')
dict4 = unpickle('CIFAR10/cifar-10-batches-py/data_batch_4')
dict5 = unpickle('CIFAR10/cifar-10-batches-py/data_batch_5')
dict6 = unpickle('CIFAR10/cifar-10-batches-py/test_batch')

# Begin Defining Train Data/Label Arrays
arrTrainData = np.concatenate((reshape_image(dict1['data']), reshape_image(dict2['data']), reshape_image(dict3[
                        'data']), reshape_image(dict4['data']), reshape_image(dict5['data'])), axis=0, dtype=int)
arrTrainLabels = np.concatenate((dict1['labels'], dict2['labels'], dict3['labels'], dict4['labels'], dict5['labels']),
                           axis=0, dtype=int)
DataTrainTensor = torch.from_numpy(arrTrainData)
LabelTrainTensor = torch.from_numpy(arrTrainLabels)

# Begin Defining Test Data/Label Arrays
arrTestData = np.array(reshape_image(dict6['data']), dtype=int)
arrTestLabels = np.array(dict6['labels'], dtype=int)
DataTestTensor = torch.from_numpy(arrTestData)
LabelTestTensor = torch.from_numpy(arrTestLabels)

train_ds = CustDataset(DataTrainTensor, LabelTrainTensor)
test_ds = CustDataset(DataTestTensor, LabelTestTensor)

train_dl = DataLoader(train_ds, batch_size=batch_size)
# train_dl2 = DataLoader(train_ds, batch_size=1)
# test_dl = DataLoader(test_ds, batch_size=batch_size)
# test_dl2 = DataLoader(test_ds, batch_size=1)

transforms1 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                 transforms.RandomRotation(30)])
transforms2 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# train_rot_dataset = torchvision.datasets.CIFAR10(root='CIFAR10', train=True, transform=transforms1, download=False)

model = ConvNN()

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
epoch_list = []
losses = []

print(f"Time to load: {(time.time() - t_start):.3f} sec")
print(f"Epochs: {epochs} | Batch: {batch_size} | Learning Rate: {learning_rate}")


epoch_list, losses = train_model(train_dl, model)
print(epoch_list)
print(losses)

print(f"Program executed in {(time.time() - t_start):.1f} sec.\n")
print("----------------------------------------------------------\n")
