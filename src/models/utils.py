
import torch
from transformers.modeling_outputs import TokenClassifierOutput
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoModel
from datetime import timedelta
import random
from torch.utils.data import Dataset
import numpy as np
import os
import shutil
import adapters



class CustomClassifier(nn.Module):
    def __init__(self, model, classification_layer):
        super(CustomClassifier, self).__init__()
        self.model = model
        self.additional = classification_layer
        
    def forward(self, input_ids = None, attention_mask=None, labels = None, token_type_ids= None ):
        x = self.model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids= token_type_ids)[0]
        if len(x.shape) == 3:
            x = x[:,0,:]
        x = self.additional(x)
        x = torch.sigmoid(x)
        if labels is not None:
            loss = torch.nn.functional.binary_cross_entropy(x.squeeze(), labels.float())
            return TokenClassifierOutput(logits=x,loss=loss)
        return TokenClassifierOutput(logits=x)


    @staticmethod
    def load(path, adapter_path=None):
        model = AutoModel.from_pretrained(path)
        additional = torch.load(path + "/classification.pth")
        if adapter_path is not None:
            adapters.init(model.roberta)
            model.model.roberta.load_adapter(adapter_path, set_active=True)
        return CustomClassifier(model,additional)
    
    def save(self,path):
        self.model.save_pretrained(path)
        torch.save(self.additional, path + "/classification.pth")




class CustomEnsemble(nn.Module):
    def __init__(self, models, configs=None, additional = None):
        super(CustomEnsemble, self).__init__()
        self.models = models
        if configs is not None:
            hidden_size = sum([config.hidden_size for config in configs])
            self.additional = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size,1)) 
        elif additional is not None:
            self.additional = additional
        else:
            raise ValueError("Either configs or additional must be provided.")

    def forward(self, input_ids = None, attention_mask=None, labels = None ):

        outputs = []
        for model in self.models:
            outputs.append(model(input_ids = input_ids, attention_mask = attention_mask)[0])
        x = torch.cat(outputs, dim=0)
        x = self.additional(x)
        x = torch.sigmoid(x)
        if labels is not None:
            loss = torch.nn.functional.binary_cross_entropy(x.squeeze(), labels.float())
            return TokenClassifierOutput(logits=x,loss=loss)
        return TokenClassifierOutput(logits=x)
    
    @staticmethod
    def load(path):
        models = []
        for subfolder in os.listdir(path):
            subfolder_path = os.path.join(path, subfolder)
            if os.path.isdir(subfolder_path):
                models.append(AutoModel.from_pretrained(subfolder_path))
        additional = torch.load(path + "/classification.pth")
        return CustomEnsemble(models=models,additional=additional)
    
    def save(self,path):
        for i, model in enumerate(self.models):
            model.save_pretrained(path +"/"+str(i))
        torch.save(self.additional, path + "/classification.pth")


def format_time(seconds):
    return str(timedelta(seconds=int(round(seconds))))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)




class TweetDataset(Dataset):
    def __init__(self, tweets, labels, tokenizer, max_length=128):
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tweet = str(self.tweets[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            tweet,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item



def create_clean_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)
