from transformers import RobertaConfig, RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
import torch.nn as nn


class RecipeModel(nn.Module):
    def __init__(self,num_labels=5):
        super().__init__()
        
        self.bert = RobertaModel.from_pretrained('roberta-base',add_pooling_layer=False)
        self.bert_config = self.bert.config
        self.bert_config.num_labels = num_labels
        
        self.head = RobertaClassificationHead(self.bert.config)

    def forward(self,**kwargs):

        x = self.bert(**kwargs)
        
        return self.head(x[0])
