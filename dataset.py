from transformers import RobertaTokenizer
from torch.utils.data import Dataset

class RecipeDataset(Dataset):

    def __init__(self,X,y) -> None:
        self.X = X
        self.y = y -1
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        text = self.X[idx]
        score = self.y[idx]
        tokenized_text = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=200,
            return_tensors='pt'
        )

        tokenized_text['input_ids'] = tokenized_text['input_ids'].squeeze(0)
        tokenized_text['attention_mask'] = tokenized_text['attention_mask'].squeeze(0)
        return tokenized_text, score
        pass
