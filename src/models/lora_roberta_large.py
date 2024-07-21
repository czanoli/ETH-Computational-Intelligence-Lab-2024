from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
import torch
from train_llms import train
from utils import *

def execute(path, validation,configfile):
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-large-topic-sentiment-latest")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-large-topic-sentiment-latest")
    additional = nn.Linear(5,1)
    additional.weight.data = torch.tensor([[-1., -1., 0., 1., 1.]], dtype=torch.float32)
    additional.bias.data = torch.tensor([0.], dtype=torch.float32)
    model = CustomClassifier(model, additional,configfile['models_lora_roberta_large'])
    model = train(path, model, tokenizer, lr = configfile['models_lora_roberta_large']['lr'], num_epochs=configfile['models_lora_roberta_large']['epochs'], validation=validation)
    tokenizer.save_pretrained("models/lora-twitter-roberta-large-topic-sentiment-latest")
    model.save("models/lora-twitter-roberta-large-topic-sentiment-latest")
    model.model.roberta.save_adapter("models/lora-twitter-roberta-large-topic-sentiment-latest/adapter", "lora_adapter")

