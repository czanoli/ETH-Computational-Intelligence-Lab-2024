from sklearn.metrics import accuracy_score
from transformers import get_scheduler
from tqdm.auto import tqdm
import torch
import time 
from utils import *

def train(train_loader, model, lr= 2e-5,num_epochs= 3, seed= 42, val_loader= None):
    set_seed(seed)
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
        if val_loader is not None:
            model.eval()
            all_labels = []
            all_preds = []
            with torch.no_grad():
                for eval_batch in val_loader:
                    eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                    outputs = model(**eval_batch)
                    preds = torch.where(outputs.logits <= 0.5, torch.tensor(0), torch.tensor(1))
                    all_labels.extend(eval_batch['labels'].cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
            
            val_accuracy = accuracy_score(all_labels, all_preds)
            print(f"Epoch {epoch + 1} took {format_time(time.time() - start_time)}. Validation Accuracy = {val_accuracy:.4f} ")
            model.train() 
    return model