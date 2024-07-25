from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
import torch.nn as nn
from train_llms import train
from utils import *

def execute(path, validation,configfile):
    set_seed(configfile['random_state'])
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    model = AutoModelForMaskedLM.from_pretrained("vinai/bertweet-base").roberta
    config = AutoConfig.from_pretrained("vinai/bertweet-base")
    additional = nn.Linear(config.hidden_size, 1)
    model = CustomClassifier(model, additional, configfile['models_bertweet_base'])
    train_loader, val_loader = get_tweets_loader(path, tokenizer, validation= validation, seed=configfile['random_state'])
    model = train(train_loader, model, lr= configfile['models_bertweet_base']['lr'], num_epochs= configfile['models_bertweet_base']['epochs'], val_loader= val_loader, seed=configfile['random_state'])
    tokenizer.save_pretrained("models/finetuned-bertweet-base")
    model.save("models/finetuned-bertweet-base")