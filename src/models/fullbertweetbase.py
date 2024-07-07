from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from train_llms import train,CustomClassifier
import torch.nn as nn


def execute(path, validation,configfile):
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    model = AutoModelForMaskedLM.from_pretrained("vinai/bertweet-base").roberta
    config = AutoConfig.from_pretrained("vinai/bertweet-base")
    additional = nn.Linear(config.hidden_size, 1)
    model = CustomClassifier(model, additional)
    model = train(path, model, tokenizer, lr = configfile['models_bertweet_base']['lr'], num_epochs=configfile['models_bertweet_base']['epochs'], validation=validation)
    tokenizer.save_pretrained("models/finetuned-bertweet-base")
    model.save("models/finetuned-bertweet-base")
