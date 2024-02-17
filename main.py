import torch

import pandas as pd

import numpy as np
import transformers
from torch.utils.data import DataLoader
from dataset import RecipeDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

#local imports
from model import RecipeModel

BATCH_SIZE=32
EPOCHS=10
LR=3e-4


df = pd.read_csv("./Dataset/preprocessed_data.csv")
data = df[["text","stars"]]
print(data.head())
X_train, X_test, y_train, y_test = train_test_split( data["text"].to_numpy()[:100], data["stars"].to_numpy()[:100], test_size=0.20, random_state=42)

train_dataset = RecipeDataset(X_train, y_train)
valid_dataset = RecipeDataset(X_test, y_test)


train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
valid_loader = DataLoader(valid_dataset,batch_size=BATCH_SIZE,shuffle=False)

rmodel = RecipeModel()

optimizer = torch.optim.Adam(params=rmodel.parameters())
criterion = torch.nn.CrossEntropyLoss()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

rmodel = rmodel.to(DEVICE)


for e in tqdm(range(EPOCHS),position=1):
    losses = []
    for batch in tqdm(train_loader,position=0):
        inp,labels = batch

        #transfer to gpu
        for key in inp:
            inp[key] = inp[key].to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        
        pred = rmodel(**inp)

        loss = criterion(pred,labels)
        
        losses.append(loss.cpu().item())
        # print(loss.cpu().item())
        loss.backward()

        optimizer.step()

    e_loss = np.array(losses).mean()
    print(f"EPOCH {e}: Loss:{e_loss}")

    val_losses = []
    for batch in tqdm(valid_loader,position=0):
        inp,labels = batch

        #transfer to gpu
        for key in inp:
            inp[key] = inp[key].to(DEVICE)
        labels = labels.to(DEVICE)
        
        pred = rmodel(**inp)

        loss = criterion(pred,labels)
        
        val_losses.append(loss.cpu().item())
        # print(loss.cpu().item())

    e_loss = np.array(val_losses).mean()
    print(f"EPOCH {e}: Val Loss:{e_loss}")

