import click
import joblib
import pandas as pd
import numpy as np
import logging
import yaml
from pathlib import Path
import subprocess
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from train_llms import CustomClassifier
import adapters
from utils import *
from torch.utils.data import DataLoader
logger = logging.getLogger(__name__)

# Load configuration from config.yml
with open(Path(__file__).resolve().parent/'config.yml', 'r') as ymlfile:
    config = yaml.safe_load(ymlfile)
BOW = config['BOW']
GLOVE = config['GLOVE']
GLOVE_VECTOR_SIZE = config['GLOVE_VECTOR_SIZE']
VECTORIZER_PATH = config['VECTORIZER_PATH']
GLOVE_PATH = config['glove_path']
RESULTS_PATH = config['RESULTS_PATH']
XTEST_PATH = config['fasttext']['xtest_path']

def load_glove_embeddings(file_path):
    """
    Load GloVe embeddings from a file and return a dictionary of word vectors.

    Parameters
    ----------
    file_path : str
        Path to the GloVe embeddings file.

    Returns
    -------
    dict
        Dictionary of word vectors.
    """
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def tweet_to_glove_vector(tweet, embeddings, vector_size=GLOVE_VECTOR_SIZE):
    """
    Convert a tweet to a GloVe vector by averaging the vectors of the words in the tweet.

    Parameters
    ----------
    tweet : str
        The tweet to convert.
    embeddings : dict
        Dictionary of word vectors.
    vector_size : int, optional
        Size of the word vectors, by default GLOVE_VECTOR_SIZE.

    Returns
    -------
    np.ndarray
        Averaged vector of the tweet.
    """
    words = tweet.lower().split()
    tweet_vec = np.zeros(vector_size)
    count = 0
    for word in words:
        if word in embeddings:
            tweet_vec += embeddings[word]
            count += 1
    if count != 0:
        tweet_vec /= count
    return tweet_vec


def create_fasttext_format(df, file_path, is_test=False):
    """
    Create a text file in the FastText format from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    file_path : str
        The path to the output file.
    is_test : bool, optional
        Flag indicating whether the file is for test data, by default False.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            if is_test:
                tweet = row['tweet'].replace('\n', ' ')
                f.write(f"{tweet}\n")
            else:
                label = "__label__" + str(row['label'])
                tweet = row['tweet'].replace('\n', ' ')
                f.write(f"{label} {tweet}\n")
    
    
def predict_classifiers(model_path, datapath, method, embedding):
    """
    Predict using a 'classifier' model on the test data.

    Parameters
    ----------
    model_path : str
        Path to the trained model.
    datapath : str
        Path to the test data.
    method : str
        Training method used.
    embedding : str
        Embedding method used.
    """
    df_test = pd.read_csv(datapath)
    model = joblib.load(model_path)
    X_test = df_test['tweet']

    # Transform the test set using the same vectorizer and make predictions
    if embedding == BOW:
        logger.info('[BoW]: Loading vectorizer...')
        vectorizer = joblib.load(VECTORIZER_PATH)
        logger.info('[BoW]: Vectorizer loaded.')
        logger.info('[BoW]: Vectorizing X_test...')
        X_test_vec = vectorizer.transform(X_test)
        logger.info('[BoW]: X_test vectorized.')
    elif embedding == GLOVE:
        logger.info('[GloVe]: Loading GloVe embeddings...')
        glove_embeddings = load_glove_embeddings(GLOVE_PATH)
        logger.info('[GloVe]: GloVe embeddings loaded.')
        logger.info('[GloVe]: Vectorizing X_test...')
        X_test_vec = np.array([tweet_to_glove_vector(tweet, glove_embeddings) for tweet in X_test])
        logger.info('[GloVe]: X_test vectorized.')

    logger.info('Making predictions...')
    y_test_pred = model.predict(X_test_vec)

    # Create the final DataFrame with Id and Prediction columns
    df_test['prediction'] = y_test_pred
    df_test['prediction'] = df_test['prediction'].replace(0, -1)
    df_final = df_test[['id', 'prediction']]
    df_final = df_final.rename(columns={'id': 'Id', 'prediction': 'Prediction'})

    # Save the final DataFrame to a CSV file
    file_path = RESULTS_PATH + f"predictions_{method}_{embedding}.csv"
    df_final.to_csv(file_path, index=False)
    logger.info(f'Predictions saved at {file_path}')

    # Print the first few rows of the final DataFrame
    print(df_final.head())
    
def predict_fasttext(modelpath, datapath):
    """
    Predict using the FastText model on the test data.

    Parameters
    ----------
    modelpath : str
        Path to the trained FastText model.
    datapath : str
        Path to the test data.
    """
    df_test = pd.read_csv(datapath)
    logger.info("Formatting test set...")
    create_fasttext_format(df_test, XTEST_PATH, is_test=True)
    logger.info("Test set formatted.")
    
    try:
        logger.info('Making predictions...')
        fasttext_command = fasttext_command = f"src/models/fastText-0.9.2/fasttext predict {modelpath} {XTEST_PATH} > results/fasttext_predictions.txt"
        subprocess.run(fasttext_command, shell=True, check=True, text=True, capture_output=True)
        logger.info('Predictions saved at results/fasttext_predictions.txt')
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during inference: {e.stderr}")

def predict_llms(model_path, data_path, configfile):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = CustomClassifier.load(model_path,configfile)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    print("Using ", device)
    df = pd.read_csv(data_path)
    inputs = tokenizer(list(df['tweet']), return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    dataset = TweetDataset(tweets=df['tweet'].tolist(), tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
      
    with torch.no_grad():
        predictions = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = np.concatenate((predictions, torch.where(outputs.logits <= 0.5, torch.tensor(-1), torch.tensor(1)).squeeze().cpu().numpy()))
 
    df_final = pd.DataFrame({
        'id': range(1, len(predictions) + 1),
        'prediction': predictions
    })
    df_final['prediction'] = df_final['prediction'].astype(int)
    df_final.to_csv('predictions.csv', index=False, sep=',')
    print("Predictions saved to predictions.csv")

@click.command()
@click.option('--model', 'model_path', type=str, required=False, help='Path to the model')
@click.option('--data', 'data_path', type=str, required=True, help='Path to the test data')
@click.option('--method', type=click.Choice(['classifiers', 'fastText', 'CNN', 'RNN','twitter-roberta-base-sentiment-latest', 'lora-roberta-large-sentiment-latest','bertweet-base','lora-bertweet-large', 'ensemble-small']), required=True, help='Method used for training')
@click.option('--embedding', type=click.Choice(['BoW', 'GloVe']), required=False, help='Embedding method to used if method is classifiers')
def main(model_path, data_path, method, embedding):
    """
    Main function to predict based on the specified method, embedding, and model.

    Parameters
    ----------
    model_path : str
        Path to the model.
    data_path : str
        Path to the test data.
    method : str
        Method used for training.
    embedding : str
        Embedding method to use if method is 'classifiers'.
    """
    if method == 'classifiers' and not embedding:
        raise click.UsageError("Argument --embedding is required when --method is 'classifiers'")
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    if method == "classifiers":
        predict_classifiers(model_path, data_path, method, embedding)
    if method == "fastText":
        predict_fasttext(model_path, data_path)
    if method == "twitter-roberta-base-sentiment-latest":
        predict_llms(model_path="models/finetuned-twitter-roberta-base-sentiment-latest", data_path=data_path, configfile=config['models_roberta_base'])
    if method == "lora-roberta-large-sentiment-latest":
        predict_llms(model_path="models/lora-twitter-roberta-large-topic-sentiment-latest", data_path=data_path, configfile=config['models_lora_roberta_large'])
    if method == "bertweet-base":
        predict_llms(model_path="models/finetuned-bertweet-base", data_path=data_path, configfile=config['models_bertweet_base'])
    if method == "lora-bertweet-large":
        predict_llms(model_path="models/lora-bertweet-large", data_path=data_path, configfile=config['models_lora_bertweet_large'])
    #if method == "ensemble-small":
        #TO DO
        
if __name__ == "__main__":
    main()
