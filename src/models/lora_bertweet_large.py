from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
import torch.nn as nn
from train_llms import train
from utils import *

def execute(path, validation,configfile):
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large")
    model = AutoModelForMaskedLM.from_pretrained("vinai/bertweet-large").roberta
    config = AutoConfig.from_pretrained("vinai/bertweet-large")
    additional = nn.Linear(config.hidden_size, 1)
    model = CustomClassifier(model, additional,configfile['models_lora_bertweet_large'])
    train_loader, val_loader = get_tweets_loader(path, tokenizer, validation= validation)
    model = train(train_loader, model, lr= configfile['models_lora_bertweet_large']['lr'], num_epochs= configfile['models_lora_bertweet_large']['epochs'], val_loader= val_loader)
    tokenizer.save_pretrained("models/lora-bertweet-large")
    model.save("models/lora-bertweet-large")
    model.model.save_adapter("models/lora-bertweet-large/adapter", "lora_adapter")