
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
    def __init__(self, model, classification_layer, configuration, adapter_config):
        super(CustomClassifier, self).__init__()
        self.model = model
        self.additional = classification_layer
        self.configuration = configuration
        self.adapter_config = adapter_config

        
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
    def load(path):
        model = AutoModel.from_pretrained(path)
        additional = torch.load(path + "/classification.pth")
        return CustomClassifier(model,additional)
    
    def save(self,path):
        self.model.save_pretrained(path)
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
    def __init__(self, tweets, tokenizer, max_length=128, labels=None):
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tweet = str(self.tweets[idx])
        
        encoding = self.tokenizer(
            tweet,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        item = {key: val.squeeze() for key, val in encoding.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def create_clean_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)
