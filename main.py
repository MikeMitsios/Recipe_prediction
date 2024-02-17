import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

import numpy as np
import transformers
from torch.utils.data import DataLoader
from dataset import RecipeDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score
from tqdm import tqdm
from utils import ordinal_convertion, custom_loss, calculate_pred

#local imports
from model import RecipeModel

BATCH_SIZE=32
EPOCHS=10
LR=6e-6
LOG_STEPS=1
loss_type = "ordinal" #mse, cross, ordinal

checkpoint_e = 1
filename = "check_mse_lr_6e_6"
extra_features = ["recipe_name", "reply_count", "thumbs_up", "thumbs_down", "user_id"]

df = pd.read_csv("./Dataset/preprocessed_data.csv")
data_features = df[["text"]+extra_features].to_numpy()
num_classes = len(df["stars"].unique())
data_stars = df[["stars"]].to_numpy().squeeze(-1) 
#print(data_features.head())
X_train, X_test, y_train, y_test = train_test_split( data_features, data_stars, test_size=0.20, random_state=42)

train_dataset = RecipeDataset(X_train, y_train,feats=['text']+extra_features)
valid_dataset = RecipeDataset(X_test, y_test)


train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
valid_loader = DataLoader(valid_dataset,batch_size=BATCH_SIZE,shuffle=False)

rmodel = RecipeModel()

optimizer = torch.optim.Adam(params=rmodel.parameters())
criterion = custom_loss  # torch.nn.CrossEntropyLoss()

writer = SummaryWriter()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

rmodel = rmodel.to(DEVICE)
rmodel.train()


for e in tqdm(range(1,EPOCHS+1),position=1):
    losses = []
    preds = []
    gt = []
    for bn,batch in tqdm(enumerate(train_loader),position=0,total=len(train_loader)):
        inp,extra_ins,labels = batch

        #transfer to gpu
        for key in inp:
            inp[key] = inp[key].to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        
        pred = rmodel(**inp)

        loss = criterion(pred,labels, loss_type)

        losses.append(loss.cpu().item())
        preds.extend(calculate_pred(pred, loss_type, num_classes))
        gt.extend(labels.cpu().tolist())
        # print(loss.cpu().item())
        loss.backward()
        
        if bn % LOG_STEPS == 0:
            writer.add_scalar('TrainIter/Loss',loss.cpu().item(),bn*e)

        optimizer.step()

    e_loss = np.array(losses).mean()
    writer.add_scalar('TrainEpoch/Loss',e_loss,e)
    f1 = f1_score(gt, preds, average='micro')
    writer.add_scalar('TrainEpoch/F1',f1,e)
    acc = accuracy_score(gt,preds)
    writer.add_scalar('TrainEpoch/Acc',acc,e)
    print(f"EPOCH {e}: Loss:{e_loss} | F1: {f1} | Acc: {acc}")

    val_losses = []
    val_preds = [] 
    val_gt = []
    for batch in tqdm(valid_loader,position=0):
        inp,extra_ins,labels = batch

        #transfer to gpu
        for key in inp:
            inp[key] = inp[key].to(DEVICE)
        labels = labels.to(DEVICE)
        
        pred = rmodel(**inp)

        loss = criterion(pred,labels, loss_type)
        
        val_losses.append(loss.cpu().item())
        val_preds.extend(calculate_pred(pred, loss_type, num_classes))
        val_gt.extend(labels.cpu().tolist())
        # print(loss.cpu().item())

    if e%checkpoint_e == 0:
        torch.save(rmodel.state_dict(),"./checkpoints/"+filename+str(e)+".pt")
    e_loss = np.array(val_losses).mean()
    writer.add_scalar('ValEpoch/Loss',e_loss,e)
    val_f1 = f1_score(val_gt, val_preds, average='micro')
    writer.add_scalar('ValEpoch/F1',val_f1,e)
    val_acc = accuracy_score(val_gt,val_preds)
    writer.add_scalar('ValEpoch/Acc',val_acc,e)
    print(f"EPOCH {e}: Val Loss:{e_loss} | Val F1: {val_f1} | Acc: {val_acc}")
