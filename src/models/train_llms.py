import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
import time 
from datetime import timedelta

def format_time(seconds):
    return str(timedelta(seconds=int(round(seconds))))

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

def train(train_path, model, tokenizer):

    df = pd.read_csv(train_path)
    df['label'] = df['label'].map({"positive": 2, "negative": 0}) 
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    train_dataset = TweetDataset(train_df['tweet'].tolist(), train_df['label'].tolist(), tokenizer)
    val_dataset = TweetDataset(val_df['tweet'].tolist(), val_df['label'].tolist(), tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    num_epochs = 5
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
            progress_bar.update(1)

        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=-1)
                all_labels.extend(batch['labels'].cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        
        val_accuracy = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch + 1} took {format_time(time.time() - start_time)} seconds. Validation Accuracy = {val_accuracy:.4f} ")
        model.train() 
    
    return model
    

