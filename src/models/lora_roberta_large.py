import numpy as np
import torch
import loralib as lora
from adapters import LoRAConfig
import adapters
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
from train_llms import train
from utils import *


def execute(path, validation,configfile):
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-large-topic-sentiment-latest")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-large-topic-sentiment-latest")
    additional = nn.Linear(5,1)
    additional.weight.data = torch.tensor([[-1., -1., 0., 1., 1.]], dtype=torch.float32)
    additional.bias.data = torch.tensor([0.], dtype=torch.float32)
    model = CustomClassifier(model, additional)
    adapters.init(model.model.roberta)
    config = LoRAConfig(selfattn_lora=True, intermediate_lora=True, output_lora=True, attn_matrices=['q','k','v'], r=8, alpha=16)
    model.model.roberta.add_adapter("lora_adapter", config=config)
    model.model.roberta.train_adapter("lora_adapter")
    model = train(path, model, tokenizer, lr = configfile['models_lora_roberta_large']['lr'], num_epochs=configfile['models_lora_roberta_large']['epochs'], validation=validation)
    tokenizer.save_pretrained("models/lora-twitter-roberta-large-topic-sentiment-latest")
    model.save("models/lora-twitter-roberta-large-topic-sentiment-latest")
    model.model.roberta.save_adapter("models/lora-twitter-roberta-large-topic-sentiment-latest/adapter", "lora_adapter")

