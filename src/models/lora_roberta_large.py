from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
import torch
from train_llms import train
from utils import *

def execute(path, validation,configfile):
    """
    Execute the fine-tuning and saving process for cardiffnlp/twitter-roberta-large-topic-sentiment-latest with LoRA adaptation and modified for binary-classification sentiment analysis.
    https://huggingface.co/cardiffnlp/twitter-roberta-large-topic-sentiment-latest

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
        The trained model is saved in models/lora-twitter-roberta-large-topic-sentiment-latest
    """
    set_seed(configfile['random_state'])
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-large-topic-sentiment-latest")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-large-topic-sentiment-latest")
    additional = nn.Linear(5,1)
    additional.weight.data = torch.tensor([[-1., -1., 0., 1., 1.]], dtype=torch.float32)
    additional.bias.data = torch.tensor([0.], dtype=torch.float32)
    model = CustomClassifier(model, additional,configfile['models_lora_roberta_large'])
    train_loader, val_loader = get_tweets_loader(path, tokenizer, validation= validation, seed=configfile['random_state'])
    model = train(train_loader, model, lr= configfile['models_lora_roberta_large']['lr'], num_epochs= configfile['models_lora_roberta_large']['epochs'], val_loader= val_loader,seed=configfile['random_state'])
    tokenizer.save_pretrained("models/lora-twitter-roberta-large-topic-sentiment-latest")
    model.save("models/lora-twitter-roberta-large-topic-sentiment-latest")
    model.model.roberta.save_adapter("models/lora-twitter-roberta-large-topic-sentiment-latest/adapter", "lora_adapter")

