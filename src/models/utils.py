from transformers import AutoModelForSequenceClassification, AutoModel
from transformers.modeling_outputs import TokenClassifierOutput
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from datetime import timedelta
import pandas as pd
import torch.nn as nn
import torch
import random
import numpy as np
import os
import shutil
import adapters
from adapters import LoRAConfig

class Ensemble(nn.Module):
    def __init__(self, hidden_size):
        super(Ensemble, self).__init__()
        self.fc1 = nn.Linear(int(hidden_size), int(hidden_size/2))
        self.af1 = nn.ReLU()
        self.fc2 = nn.Linear(int(hidden_size/2),1)

    def forward(self, embeddings, labels= None):
        x = self.fc1(embeddings)
        x = self.af1(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        if labels is not None:
            loss = torch.nn.functional.binary_cross_entropy(x.squeeze(), labels.float())
            return TokenClassifierOutput(logits=x,loss=loss)
        return TokenClassifierOutput(logits=x)

class CustomClassifier(nn.Module):
    def __init__(self, model, classification_layer, configfile, adapter_path = None):
        super(CustomClassifier, self).__init__()
        self.model = model
        self.additional = classification_layer
        self.config = configfile
        self.encoder_model = self.model
        encoder_path = self.config['encoder']
        if encoder_path != '':
            attributes = encoder_path.split('.')
            for attr in attributes:
                self.encoder_model = getattr(self.encoder_model,attr)
        if self.config['lora']:
            adapters.init(self.encoder_model)
            if adapter_path is not None:
                self.encoder_model.load_adapter(adapter_path)
            else:
                lora_config = LoRAConfig(selfattn_lora=self.config['selfattn_lora'], intermediate_lora=self.config['intermediate_lora'], output_lora=self.config['output_lora'], attn_matrices=self.config['attn_matrices'], r=self.config['r'], alpha=self.config['alpha'])
                self.encoder_model.add_adapter('lora_adapter', config=lora_config)
            self.encoder_model.set_active_adapters('lora_adapter')
            self.encoder_model.train_adapter("lora_adapter")

        
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
    def load(path, configfile):
        if configfile['isForClassification']:
            model = AutoModelForSequenceClassification.from_pretrained(path)
        else:
            model = AutoModel.from_pretrained(path)
        additional = torch.load(path + "/classification.pth")
        return CustomClassifier(model, additional, configfile, path + "/adapter")
    
    def save(self,path):
        self.model.save_pretrained(path)
        torch.save(self.additional, path + "/classification.pth")
        if self.config['lora']:
            self.encoder_model.save_adapter(path + "/adapter", 'lora_adapter')

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
    
class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings, labels= None):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        item = {"embeddings": self.embeddings[idx]}
        if self.labels is None:
            return item
        item["labels"] = self.labels[idx]
        return item

def get_tweets_loader(path, tokenizer, seed= 42, validation= False):
    df = pd.read_csv(path)
    df['label'] = df['label'].map({"positive": 1, "negative": 0}) 
    if validation:
        train_df, val_df = train_test_split(df, test_size= 0.1, random_state= seed)
        val_dataset = TweetDataset(tweets= val_df['tweet'].tolist(), labels= val_df['label'].tolist(), tokenizer= tokenizer)
        val_loader = DataLoader(val_dataset, batch_size= 32, shuffle= False, pin_memory= True)
    else:
        train_df = df
    train_dataset = TweetDataset(tweets= train_df['tweet'].tolist(), labels= train_df['label'].tolist(), tokenizer= tokenizer)
    train_loader = DataLoader(train_dataset, batch_size= 32, shuffle= True, pin_memory= True)
    if validation:
        return train_loader, val_loader
    return train_loader, None

def get_embeddings_loader(embeddings_paths, labels_path= None, seed= 42, validation= False):
    if labels_path is not None:
        X, y = couple_data(embeddings_paths,labels_path)
        if validation:
            X, X_val, y, y_val = train_test_split(X, y, test_size= 0.1, random_state= seed, shuffle= True)
            val_dataset = EmbeddingsDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size= 32, shuffle= False, pin_memory= True)
        train_dataset = EmbeddingsDataset(X,y)
        train_loader = DataLoader(train_dataset, batch_size= 32, shuffle= True, pin_memory= True)
        if validation:
            return train_loader, val_loader
        return train_loader, None
    X = couple_data(embeddings_paths)
    test_dataset = EmbeddingsDataset(X)
    test_loader = DataLoader(test_dataset, batch_size= 32, shuffle= False, pin_memory= True)
    return test_loader

def couple_data(embedding_paths, labels_path= None):
    list_embeddings = []
    for curr in embedding_paths:
        list_embeddings.append(np.array(torch.load(curr)))
    embeddings = np.concatenate(list_embeddings, axis=1)
    if labels_path is not None:
        df = pd.read_csv(labels_path)
        labels = df['label'].map({"positive": 1, "negative": 0}).to_list()
        return embeddings, labels
    return embeddings

def format_time(seconds):
    return str(timedelta(seconds=int(round(seconds))))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_clean_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)

def save_predictions(y_pred):
    df_final = pd.DataFrame({
        'id': range(1, len(y_pred) + 1),
        'prediction': y_pred
    })
    df_final['prediction'] = df_final['prediction'].astype(int)
    df_final.to_csv('predictions.csv', index=False, sep=',')
    print("Predictions saved to predictions.csv")
