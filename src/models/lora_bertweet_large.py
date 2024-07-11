from adapters import LoRAConfig
import adapters
from transformers import AutoTokenizer
from train_llms import train
from utils import *
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from train_llms import train,CustomClassifier
import torch.nn as nn


def execute(path, validation,configfile):
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large")
    model = AutoModelForMaskedLM.from_pretrained("vinai/bertweet-large").roberta
    config = AutoConfig.from_pretrained("vinai/bertweet-large")
    additional = nn.Linear(config.hidden_size, 1)
    model = CustomClassifier(model, additional)
    adapters.init(model.model)
    config = LoRAConfig(selfattn_lora=True, intermediate_lora=True, output_lora=True, attn_matrices=['q','k','v'], r=8, alpha=16)
    model.model.add_adapter("lora_adapter", config=config)
    model.model.train_adapter("lora_adapter")
    model = train(path, model, tokenizer, lr = configfile['models_lora_bertweet_large']['lr'], num_epochs=configfile['models_lora_bertweet_large']['epochs'], validation=validation)
    tokenizer.save_pretrained("models/lora-bertweet-large")
    model.save("models/lora-bertweet-large")
    model.model.save_adapter("models/lora-bertweet-large/adapter", "lora_adapter")
