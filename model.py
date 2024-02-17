from transformers import RobertaConfig, RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
import torch.nn as nn
import torch


class RecipeModel(nn.Module):
    def __init__(self,num_labels=5,num_users=14000):
        super().__init__()

        inter_dim = 16
        self.num_users = num_users 
        self.bert = RobertaModel.from_pretrained('roberta-base',add_pooling_layer=False)
        self.bert_config = self.bert.config
        self.bert_config.num_labels = inter_dim
        
        self.head = RobertaClassificationHead(self.bert.config)

        self.user_embedding = nn.Embedding(num_users,16)
        
        self.aux_head = nn.Sequential(nn.Linear(3+16,128),nn.ReLU(),nn.Linear(128,inter_dim))

        self.cls_head = nn.Linear(2*inter_dim,num_labels)

    def forward(self,input_ids,attention_mask,**kwargs):
        x = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        x_bert = self.head(x[0])

        rc = kwargs['reply_count'].unsqueeze(-1)
        tu = kwargs['thumbs_up'].unsqueeze(-1)
        td = kwargs['thumbs_down'].unsqueeze(-1)
        user_emb = self.user_embedding(kwargs['user_id'])

        total_feat = torch.cat([rc,tu,td,user_emb],dim=-1)
        x_other = self.aux_head(total_feat)

        total_x = torch.cat([x_bert,x_other],dim=-1)

        return self.cls_head(total_x)
    
    def inference(self,input_ids,attention_mask,**kwargs):
        x = self.bert(input_ids=input_ids,attention_mask=attention_mask)

        return self.head(x[0]).argmax(-1)


