import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time 
from tqdm.auto import tqdm
from datetime import timedelta
import random
import numpy as np
from transformers.modeling_outputs import TokenClassifierOutput
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class CustomClassifier(nn.Module):
    def __init__(self, model, classification_layer):
        super(CustomClassifier, self).__init__()
        self.model = model
        self.additional = classification_layer
        
    def forward(self, input_ids = None, attention_mask=None, labels = None ):
        x = self.model(input_ids = input_ids, attention_mask = attention_mask)[0]
        x = self.additional(x)
        x = torch.sigmoid(x)
        if labels is not None:
            loss = torch.nn.functional.binary_cross_entropy(x.squeeze(), labels.float())
            return TokenClassifierOutput(logits=x,loss=loss)
        return TokenClassifierOutput(logits=x)

    @staticmethod
    def load(path):
        model = AutoModelForSequenceClassification.from_pretrained(path)
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

def train(train_path, model, tokenizer, lr = 2e-5,num_epochs=3, seed=42, validation = False):

    set_seed(seed)
    df = pd.read_csv(train_path)
    df['label'] = df['label'].map({"positive": 1, "negative": 0}) 
    if validation:
        train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
        val_dataset = TweetDataset(val_df['tweet'].tolist(), val_df['label'].tolist(), tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    else:
        train_df = df
    
    train_dataset = TweetDataset(train_df['tweet'].tolist(), train_df['label'].tolist(), tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    print("Using ", device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        start_time = time.time()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            optimizer.zero_grad()
            progress_bar.update(1)
        if validation:
            model.eval()
            all_labels = []
            all_preds = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    preds = torch.where(outputs.logits <= 0.5, torch.tensor(0), torch.tensor(1))
                    all_labels.extend(batch['labels'].cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
            
            val_accuracy = accuracy_score(all_labels, all_preds)
            print(f"Epoch {epoch + 1} took {format_time(time.time() - start_time)}. Validation Accuracy = {val_accuracy:.4f} ")
            model.train() 
    return model
    

