import torch

import pandas as pd

import numpy as np
import transformers
from torch.utils.data import DataLoader
from dataset import RecipeDataset
from sklearn.model_selection import train_test_split


#local imports
from model import RecipeModel

df = pd.read_csv("./Dataset/preprocessed_data.csv")
data = df[["text","stars"]]
print(data.head())
X_train, X_test, y_train, y_test = train_test_split( data["text"].to_numpy(), data["stars"].to_numpy(), test_size=0.20, random_state=42)

data = RecipeDataset(X_train, y_train)


train_loader = DataLoader(data,batch_size=4,shuffle=True)

rmodel = RecipeModel()

optimizer = torch.optim.Adam(params=rmodel.parameters())
criterion = torch.nn.CrossEntropyLoss()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

rmodel = rmodel.to(DEVICE)

for batch in train_loader:
    inp,labels = batch

    #transfer to gpu
    for key in inp:
        inp[key] = inp[key].to(DEVICE)
    labels = labels.to(DEVICE)

    optimizer.zero_grad()
    
    pred = rmodel(**inp)

    loss = criterion(pred,labels)
    

    print(loss.cpu().item())
    loss.backward()

    optimizer.step()


