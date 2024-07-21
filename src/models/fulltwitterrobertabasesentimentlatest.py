from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
import torch
from train_llms import train
from utils import *

def execute(path, validation,configfile):
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    additional = nn.Linear(3,1)
    additional.weight.data = torch.tensor([[-1., 0., 1.]], dtype=torch.float32)
    additional.bias.data = torch.tensor([0.], dtype=torch.float32)
    model = CustomClassifier(model, additional,configfile['models_roberta_base'])
    model = train(path, model, tokenizer, lr = configfile['models_roberta_base']['lr'], num_epochs=configfile['models_roberta_base']['epochs'], validation=validation)
    tokenizer.save_pretrained("models/finetuned-twitter-roberta-base-sentiment-latest")
    model.save("models/finetuned-twitter-roberta-base-sentiment-latest")