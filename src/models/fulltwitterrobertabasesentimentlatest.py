from transformers import AutoTokenizer, AutoModelForSequenceClassification
from train_llms import train


def execute(path):
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    model = train(path, model, tokenizer)
    model.save_pretrained("models/finetuned-twitter-roberta-base-sentiment-latest")
    tokenizer.save_pretrained("models/finetuned-twitter-roberta-base-sentiment-latest")
