import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('task 1.csv')
data = data.drop(columns=['location', 'sites', 'time', 'date'])

# Convert categorical variables to numerical using LabelEncoder
le = LabelEncoder()
data['browser'] = le.fit_transform(data['browser'])
data['os'] = le.fit_transform(data['os'])
data['locale'] = le.fit_transform(data['locale'])

# One-hot encode categorical variables
ohe = OneHotEncoder()
categorical_vars = ['browser', 'os', 'locale']
encoded_vars = ohe.fit_transform(data[categorical_vars]).toarray()
encoded_vars = pd.DataFrame(encoded_vars)
data = pd.concat([data, encoded_vars], axis=1)
data = data.drop(columns=categorical_vars)
data.shape
# Split the data into train and test sets
y = data['user_id'].apply(lambda x: 1 if x == 0 else 0)
X = data.drop(columns=['user_id'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tenzors
X_train = torch.tensor(X_train.values.astype(np.float32))
X_test = torch.tensor(X_test.values.astype(np.float32))
y_train = torch.from_numpy(y_train.values)
y_test = torch.from_numpy(y_test.values)
X_train
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

# We work with a group of samples, i.e. batches, instead of single images.
# Usually batch_size is some power of 2.
# The bigger batch_size accelerates the training, but requires more memory
batch_size = 32
test_batch_size = 32

dataset_train = TensorDataset(X_train, y_train)
dataset_test = TensorDataset(X_test, y_test)

train_loader = DataLoader(dataset_train,
                          batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset_test,
                         batch_size=test_batch_size, shuffle=True)

print(dataset_train)

# Specify the hardware for model run, choose GPU if possible
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#     My ANN model has 3 fully connected layers with 6 input features,
# 32 hidden units in the first layer, 16 hidden units in the second layer,
# and 1 output unit with a sigmoid activation function.

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(36, 3)
        self.fc2 = nn.Linear(3, 3)
        self.fc3 = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


model = Net().to(device)

model

# Number of iterations over the whole data set
epochs = 100
# Learning rate for Stochastic Gradient Descent
lr = 0.01
# SGD parameter to accelerate the optimization, check https://paperswithcode.com/method/sgd-with-momentum
momentum = 0.5


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    test_loss = 0
    correct = 0
    for data, target in train_loader:
        target = target.view(-1, 1)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print("target: ", target, "output:", output)
        loss = criterion(output, target.float())
        loss.backward()
        optimizer.step()

        test_loss += loss
        pred = torch.round(output)

        with torch.no_grad():
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(
        f"Test set: Average loss: {test_loss / len(train_loader.dataset)}, \n  Accuracy: {100. * (correct / len(train_loader.dataset))}, \n  Correct: {correct},   Size of dataset: {len(train_loader.dataset)};  \n\n")


# Collection of optimizers
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=lr)
# Loss function
criterion = nn.BCELoss()
# bar = tqdm(train_loader)
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch, criterion)
    test(model, device, test_loader, criterion)


