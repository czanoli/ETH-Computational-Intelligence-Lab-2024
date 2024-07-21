import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time 
from tqdm.auto import tqdm
import numpy as np
from utils import *



def train(train_path, model, tokenizer, lr = 2e-5,num_epochs=3, seed=42, validation = False):

    set_seed(seed)
    df = pd.read_csv(train_path)
    df['label'] = df['label'].map({"positive": 1, "negative": 0}) 
    if validation:
        train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
        val_dataset = TweetDataset(tweets=val_df['tweet'].tolist(), labels=val_df['label'].tolist(), tokenizer=tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    else:
        train_df = df
    
    train_dataset = TweetDataset(tweets=train_df['tweet'].tolist(), labels=train_df['label'].tolist(), tokenizer=tokenizer)
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
    

