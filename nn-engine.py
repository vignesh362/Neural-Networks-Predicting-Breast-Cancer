# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import Dataset, DataLoader

#pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118


# 3 layer Neural Network
input_layer_nodes = 30
hidden_layer_nodes= 60
number_hidden_layers=2
output_layer_nodes = 1


learning_rate = 0.01
batch_size = 70

class tensorData(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:

      self.X = torch.from_numpy(X.astype(np.float32))
      self.y = torch.from_numpy(y.astype(np.float32))
      self.len = self.X.shape[0]
    def __getitem__(self, index: int) -> tuple:
      return self.X[index], self.y[index]
    def __len__(self) -> int:
      return self.len


class neuralNet(nn.Module):
    def __init__(self,input_nd,hidden_nd,output_nd):
        super(neuralNet, self).__init__()
        self.i=nn.Linear(in_features=input_nd, out_features=hidden_nd)
        self.l1=nn.Linear(in_features=hidden_nd, out_features=hidden_nd)
        self.l2=nn.Linear(in_features=hidden_nd, out_features=output_nd)
    def forward(self, x: torch.Tensor):
        x = nn.Sigmoid()(self.i(x))
        x = nn.Sigmoid()(self.l1(x))
        x = nn.ReLU()(self.l2(x))
        return x


#Features---> id,diagnosis,radius_1ean,texture_1ean,peri1eter_1ean,area_1ean,s1oothness_1ean,co1pactness_1ean,concavity_1ean,concave points_1ean,sy11etry_1ean,fractal_di1ension_1ean,radius_se,texture_se,peri1eter_se,area_se,s1oothness_se,co1pactness_se,concavity_se,concave points_se,sy11etry_se,fractal_di1ension_se,radius_worst,texture_worst,peri1eter_worst,area_worst,s1oothness_worst,co1pactness_worst,concavity_worst,concave points_worst,sy11etry_worst,fractal_di1ension_worst

file="/content/breast-cancer.csv"
dataset=np.loadtxt(file,delimiter=",", dtype=float)
x=dataset[1:,2:]
y=dataset[1:,1]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

#plt.xticks(rotation=90)
#plt.scatter(np.array(x), np.array(y))
#plt.show()
#print(nn.__file__)
trainData=tensorData(X_train,y_train)
dataLoaderTrain = DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True)

testData=tensorData(X_test,y_test)
dataLoaderTest = DataLoader(dataset=testData, batch_size=batch_size, shuffle=True)

model=neuralNet(X_train.shape[1],60,2)
print(model)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# start training
epochs = 1000
for epoch in range(epochs):
  running_loss = 0.0
  for i, (inputs, target) in enumerate(dataLoaderTrain):

    # forward propagation
    outputs = model(inputs)
    if len(target.shape) == 1:
        target = target.unsqueeze(1)
    loss = criterion(outputs, target)

    # to remove previous epoch gradients, so nxt rum will be perfect
    optimizer.zero_grad()

    # backward propagation
    loss.backward()

    # optimize
    optimizer.step()
    running_loss += loss.item()

  # display statistics
  if not ((epoch + 1) % (epochs // 10)):
    print(f'Loss: {running_loss / (i + 1):.10f}')

with torch.no_grad():
  loss = 0
  for i, (inputs, target) in enumerate(dataLoaderTest):
    predictions = model(inputs)
    print(predictions)
    print(target)
    if len(target.shape) == 1:
      target = target.unsqueeze(1)
    loss += criterion(predictions, target)
  print(loss)