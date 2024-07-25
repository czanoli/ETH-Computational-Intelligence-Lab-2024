from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
import torch.nn as nn
from train_llms import train
from utils import *

def execute(path, validation,configfile):
    """
    Execute the fine-tuning and saving process for vinai/bertweet-large with LoRA adaptation and modified for binary-classification sentiment analysis.
    https://huggingface.co/vinai/bertweet-large

    Parameters
    ----------
    path : str
        Path to the dataset for training and validation.
    validation : bool
        Flag to indicate whether to perform validation during the training epochs.
    configfile : dict
        Configuration parameters loaded from a .yml file.

    Returns
    -------
    None
        The trained model is saved in models/lora-bertweet-large
    """
    set_seed(configfile['random_state'])
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large")
    model = AutoModelForMaskedLM.from_pretrained("vinai/bertweet-large").roberta
    config = AutoConfig.from_pretrained("vinai/bertweet-large")
    additional = nn.Linear(config.hidden_size, 1)
    model = CustomClassifier(model, additional,configfile['models_lora_bertweet_large'])
    train_loader, val_loader = get_tweets_loader(path, tokenizer, validation= validation,seed=configfile['random_state'])
    model = train(train_loader, model, lr= configfile['models_lora_bertweet_large']['lr'], num_epochs= configfile['models_lora_bertweet_large']['epochs'], val_loader= val_loader, seed=configfile['random_state'])
    tokenizer.save_pretrained("models/lora-bertweet-large")
    model.save("models/lora-bertweet-large")
    model.model.save_adapter("models/lora-bertweet-large/adapter", "lora_adapter")