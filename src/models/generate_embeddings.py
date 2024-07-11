
from transformers import AutoTokenizer
import torch
from train_llms import CustomClassifier
import adapters
from utils import *
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import click
import yaml
from pathlib import Path


@click.command()
@click.option('--model_path', type=str, required=True)
@click.option('--data_path', type=str, required=True)
@click.option('--model_name', type=str, required=True)
def main(model_path, data_path, model_name):
    with open(Path(__file__).resolve().parent/'config.yml', 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = CustomClassifier.load(model_path, config[model_name]).encoder_model
    print(model)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    print("Using ", device)
    df = pd.read_csv(data_path)
    inputs = tokenizer(list(df['tweet'].astype(str)), return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    dataset = TweetDataset(tweets=df['tweet'].tolist(), tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
      
    with torch.no_grad():
        embeddings = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)[0][:,0,:]
            embeddings.extend(outputs.cpu())
    torch.save(embeddings, "data/embeddings/" +model_path.split("/")[-1] +"_"+ data_path.split("/")[-1].split(".")[0] + ".pt") 

if __name__ == "__main__":
    main()
