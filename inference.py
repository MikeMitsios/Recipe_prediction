import torch 
from model import RecipeModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from dataset import RecipeDataset
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm

BATCH_SIZE = 10
chkp_name='check9.pt'

df = pd.read_csv("./Dataset/preprocessed_data.csv")
data = df[["text","stars"]]
print(data.head())
X_train, X_test, y_train, y_test = train_test_split( data["text"].to_numpy()[:100], data["stars"].to_numpy()[:100], test_size=0.20, random_state=42)

valid_dataset = RecipeDataset(X_test, y_test)
valid_loader = DataLoader(valid_dataset,batch_size=BATCH_SIZE,shuffle=False)

rmodel = RecipeModel()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

rmodel = rmodel.to(DEVICE)


#load model
chkpt = torch.load(os.path.join('./','checkpoints',chkp_name))
rmodel.load_state_dict(chkpt)
rmodel.eval()


preds = []
gt = []

for batch in tqdm(valid_loader,position=0):
    inp,extra_ins,labels = batch

    #transfer to gpu
    for key in inp:
        inp[key] = inp[key].to(DEVICE)
    labels = labels.to(DEVICE)
    
    pred = rmodel.inference(**inp)

    preds.extend(pred.tolist())
    gt.extend(labels.tolist())

    # val_losses.append(loss.cpu().item())
    # print(loss.cpu().item())

print(classification_report(np.array(gt),np.array(preds)))