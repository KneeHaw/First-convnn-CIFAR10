import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import math
import time

t_start = time.time()

# Declare GPU:
device = torch.device("cuda:0")
# if torch.cuda.is_available() else "cpu"

# Hyper parameters
epochs = 10
batch_size = 4
learning_rate = 0.01

transforms1 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                 transforms.RandomRotation(30)])
transforms2 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_rot_dataset = torchvision.datasets.CIFAR10(root='CIFAR10', train=True, transform=transforms1, download=False)
train_dataset = torchvision.datasets.CIFAR10(root='CIFAR10', train=True, transform=transforms2, download=False)
train_final_set = torch.utils.data.ConcatDataset([train_dataset, train_rot_dataset])
test_dataset = torchvision.datasets.CIFAR10(root='CIFAR10', train=False, transform=transforms2, download=False)

train_loader = torch.utils.data.DataLoader(train_final_set, batch_size=5, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=5, shuffle=False)
print(len(test_loader))
print(len(train_loader))
# input image of 32 * 32 pixels with 3 color channels (RGB)


class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.poolk = nn.AvgPool2d(2, 2, padding=1)
        self.poolk2 = nn.AvgPool2d(2, 1, padding=1)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(9, 12, 3)
        self.conv4 = nn.Conv2d(12, 15, 3)
        self.fullcon1 = nn.Linear(16*5*5, 120)
        self.fullcon2 = nn.Linear(120, 84)
        self.fullcon3 = nn.Linear(84, 10)

    def forward(self, x):
        # print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        # x = self.poolk2(F.relu(self.conv3(x)))
        # print(x.shape)
        # x = self.poolk2(F.relu(self.conv4(x)))
        # print(x.shape)
        x = x.view(-1, 16*5*5)
        # print(x.shape)
        x = F.relu(self.fullcon1(x))
        x = torch.sigmoid(self.fullcon2(x))
        x = self.fullcon3(x)
        return x


model = ConvNN().to(device)

L = nn.CrossEntropyLoss().to(device)
opt = torch.optim.SGD(model.parameters(), lr=learning_rate)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
epoch_list = []
losses = []

print(f"Time to load: {(time.time() - t_start):.3f} sec")

print(f"Epochs: {epochs} | Batch: {batch_size} | Learning Rate: {learning_rate}")
def train_model(dl, model):
    n = len(dl)
    for epoch in range(epochs):
        t_lap = time.time()
        for i, (images, labels) in enumerate(dl):
            # input = 4 images, 3 color values, 32 x 32 pixels
            images, labels = images.to(device), labels.to(device)

            # Onward, pass it forward!
            outputs = model(images)
            loss = L(outputs, labels)

            # Optimize
            opt.zero_grad()
            loss.backward()
            opt.step()

        epoch_list.append(epoch + 1)
        losses.append(loss.item())
        print(f'Epoch: {epoch + 1} | Loss: {loss.item():.3f} | Time: {(time.time() - t_lap):.1f} sec')

    return np.array(epoch_list), np.array(losses)


epoch_list, losses = train_model(train_loader, model)

print(epoch_list)
print(losses)


def accuracy_test(dl):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(10)]
        n_class_sampled = [0 for i in range(10)]
        for images, labels in dl:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            y, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(batch_size):
                label = labels[i]
                pred = predicted[i]
                if label == pred:
                    n_class_correct[label] += 1
                n_class_sampled[label] += 1

        accuracy = 100 * n_correct / n_samples
        print(f'The network had accuracy: {accuracy}%')

        for i in range(10):
            accuracy = 100* n_class_correct[i] / n_class_sampled[i]
            print(f'Accuracy of {classes[i]}: {accuracy}%')


print("Accuracy test with training data: ")
accuracy_test(train_loader)
print("Accuracy test with test data: ")
accuracy_test(test_loader)

print(f"Program executed in {(time.time() - t_start):.1f} sec.\n")
print("----------------------------------------------------------\n")
