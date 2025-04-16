#this script is the main neural network script for this project. To begin with we will use a simple MLP architecture
#First load the data and separate it into 7 days chunks. Can use one hot encoding to add a month feature. Maybe start without time information. There may be enough info contained in just the weather data.

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,random_split
from torch.optim import Adam

# Visualization tools
import torchvision
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

filename = '%PLACEHOLDER%'
n_days=3

def data_loader(filename):
    data = pd.read_csv(filename)
    data['label']=data['Ticks']
    data.loc[data['Ticks']>1, 'label']=1 #add label which is 0 for 0 ticks (wet) and 1 for at least one tick (dry)
    data = data.drop(columns=['Ticks','date'])
    #labels=data.drop([0,1,2,3,4,5]).pop('label')
    #data = data.drop(columns=['label'])
    norm_data = data/data.max() #normalise the data
    #print(labels)
    # print(norm_data)
    big_data_list=[]
    for i in range(n_days-1,len(data)): #create a rank 3 tensor. For each row there is an associated matrix, the rows of which are the last 7days of weather and their label. The last element of the first row of this matrix is the associated label
        small_list=[]
        for j in range(n_days):
            row_day=norm_data.iloc[i-j] #picks the data for day one
            small_list.append(row_day) #append that data into a list and then repeat for the previous 7days
        big_data_list.append(small_list) #append these 7x97 matrices into a list
    day7_array=np.array(big_data_list) #resulting numpy array [list of dates x 7days x data per day]
    print(day7_array.shape)
    day7_array =day7_array[~np.isnan(day7_array).any(axis=(1,2))] #get rid of any block of 7days that has a NaN value in it.
    labels=[]
    for i in range(day7_array.shape[0]): #create list of labels. Also go through each row and delete last entry (the label)
        labeli=day7_array[i,0,96]
        labels.append(labeli)
    labels_array=np.array(labels) #list of labels in a np array
    pd_labels=pd.DataFrame(labels_array)
    day7_array=np.delete(day7_array,96,axis=2) #remove the 97th column for each 7x97 matrix aka remove label
    day7_2d=day7_array.reshape(day7_array.shape[0],-1) #flatten array so I now have a [days x features] matrix where dim(features) = 7days * number of daily features (which is 96)
    print(day7_2d.shape)
    pd_day7=pd.DataFrame(day7_2d) #returns a 2d dataframe
    pd_day7['label']=pd_labels
    return pd_day7

test=data_loader(filename)
test.to_csv('learning_data_' + str(n_days) +'.csv')



pd_day7=pd.read_csv('learning_data_'+ str(n_days) + '.csv').drop(columns='Unnamed: 0')
day7_train,day7_valid,day7_test=np.split(pd_day7.sample(frac=1,random_state=42),[int(0.6*len(pd_day7)),int(0.8*len(pd_day7))])
y_train1=day7_train.pop('label')
y_valid1=day7_valid.pop('label')
y_test1=day7_test.pop('label')
y_train=y_train1.values
y_valid=y_valid1.values
y_test=y_test1.values
x_train=day7_train.values
x_valid=day7_valid.values
x_test=day7_test.values
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyDataset(Dataset):
    def __init__(self, x_df, y_df):
        self.xs = torch.tensor(x_df).float().to(device)
        self.ys = torch.tensor(y_df).long().to(device)

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        return x, y

    def __len__(self):
        return len(self.xs)

BATCH_SIZE = 32

train_data = MyDataset(x_train, y_train)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
train_N = len(train_loader.dataset)


valid_data = MyDataset(x_valid, y_valid)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)
valid_N = len(valid_loader.dataset)

input_size = n_days*96
n_classes = 2

layers = [
    nn.Flatten(),
    nn.Linear(input_size, 512),  # Input
    nn.ReLU(),  # Activation for input
    nn.Linear(512, 124),  # Hidden
    nn.ReLU(),  # Activation for hidden
    nn.Linear(124, n_classes)  # Output
]

model = nn.Sequential(*layers)
model = torch.compile(model.to(device))

loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

def train():
    loss = 0
    accuracy = 0

    model.train()
    for x, y in train_loader:
        output = model(x)
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)
    print('Train - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

def validate():
    loss = 0
    accuracy = 0

    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            output = model(x)

            loss += loss_function(output, y).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
    print('Valid - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N

epochs = 10

for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    train()
    validate()
