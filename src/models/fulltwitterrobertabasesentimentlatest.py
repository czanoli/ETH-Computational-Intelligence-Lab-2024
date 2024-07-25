from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
import torch
from train_llms import train
from utils import *

def execute(path, validation,configfile):
    """
    Execute the full fine-tuning and saving process for cardiffnlp/twitter-roberta-base-sentiment-latest modified for binary-classification sentiment analysis.
    https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest

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
        The trained model is saved in models/finetuned-twitter-roberta-base-sentiment-latest
    """ 
    set_seed(configfile['random_state'])
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    additional = nn.Linear(3,1)
    additional.weight.data = torch.tensor([[-1., 0., 1.]], dtype=torch.float32)
    additional.bias.data = torch.tensor([0.], dtype=torch.float32)
    model = CustomClassifier(model, additional,configfile['models_roberta_base'])
    train_loader, val_loader = get_tweets_loader(path, tokenizer, validation= validation, seed=configfile['random_state'])
    model = train(train_loader, model, lr= configfile['models_roberta_base']['lr'], num_epochs= configfile['models_roberta_base']['epochs'], val_loader= val_loader,seed=configfile['random_state'])
    tokenizer.save_pretrained("models/finetuned-twitter-roberta-base-sentiment-latest")
    model.save("models/finetuned-twitter-roberta-base-sentiment-latest")