from transformers import RobertaTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm

class RecipeDataset(Dataset):

    def __init__(self,X,y,feats=['text']) -> None:
        self.all_feats = X
        self.feats = feats
        self.y = y -1

        self.text = X[:,0]
        self.recipe_names = X[:,1]
        self.reply_count = X[:,2]
        self.thumbs_up = X[:,3]
        self.thumbs_down = X[:,4]
        self.user_id = X[:,5]

        self.X = self.text

        if 'recipe_name' in feats:
            self.X = self.recipe_names + " : " + self.text
        
        if 'user_id' in feats:
            self.user2idx = {}

            print("Enumerating users")
            count = 0
            for usr in tqdm(self.user_id):
                if usr not in self.user2idx:
                    self.user2idx[usr] = count
                    count += 1
            print("Done")
            self.num_users = count

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
        extra_ins = {}
        if 'reply_count' in self.feats:
            extra_ins['reply_count'] = self.reply_count[idx]

        if 'thumbs_up' in self.feats:
            extra_ins['thumbs_up'] = self.thumbs_up[idx]

        if 'thumbs_down' in self.feats:
            extra_ins['thumbs_down'] = self.thumbs_down[idx]
        
        if 'user_id' in self.feats:
            extra_ins['user_id'] = self.user2idx[self.user_id[idx]]


        tokenized_text['input_ids'] = tokenized_text['input_ids'].squeeze(0)
        tokenized_text['attention_mask'] = tokenized_text['attention_mask'].squeeze(0)
        return tokenized_text, extra_ins, score
       
