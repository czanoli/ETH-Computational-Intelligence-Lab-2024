import click
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from scipy.sparse import vstack
import time
from datetime import timedelta
import warnings
import logging
import yaml
from pathlib import Path
import fulltwitterrobertabasesentimentlatest
import lora_roberta_large
import fullbertweetbase
import lora_bertweet_large
import ensembles
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from cnn_model import CNN
from cnn_lstm_model import CNN_LSTM
from lstm_cnn_model import LSTM_CNN
warnings.filterwarnings("ignore", category=ConvergenceWarning)
logger = logging.getLogger(__name__)

# Load configuration from config.yml
with open(Path(__file__).resolve().parent/'config.yml', 'r') as ymlfile:
    config = yaml.safe_load(ymlfile)

# Process hidden_layer_sizes for MLPClassifier hyperparameters
for hps in config['models_hparams']:
    if 'hidden_layer_sizes' in hps:
        hps['hidden_layer_sizes'] = [tuple(size) for size in hps['hidden_layer_sizes']]     
for hps in config['best_hparams']:
    if 'hidden_layer_sizes' in hps:
        hps['hidden_layer_sizes'] = [tuple(size) for size in hps['hidden_layer_sizes']]

# Set constants from config
RANDOM_STATE = config['random_state']
GLOVE_PATH = config['glove_path']
BOW = config['BOW']
GLOVE = config['GLOVE']
VAL_SIZE = config['VAL_SIZE']
CV = config['CV']
SCORING = config['SCORING']
MODEL_NAMES = config['models_names']
MODEL_HPARAMS = config['models_hparams']
OUTPUT_PATH = config['OUTPUT_PATH']
XTRAIN_PATH = config['fasttext']['xtrain_path']
XVAL_PATH = config['fasttext']['xval_path']
FASTTEXT_PATH = config['fasttext']['folder_path']
GLOVE_VECTOR_SIZE = config['GLOVE_VECTOR_SIZE']

# Define models with best hyperparameters
models = [
    LogisticRegression(random_state=RANDOM_STATE, C=config['best_hparams'][0]['C'], 
                       solver=config['best_hparams'][0]['solver']),
    LinearSVC(random_state=RANDOM_STATE, C=config['best_hparams'][1]['C'], 
              loss=config['best_hparams'][1]['loss']),
    RidgeClassifier(random_state=RANDOM_STATE, alpha=config['best_hparams'][2]['alpha']),
    SGDClassifier(random_state=RANDOM_STATE, loss=config['best_hparams'][3]['loss'], 
                  alpha=config['best_hparams'][3]['alpha'], penalty=config['best_hparams'][3]['penalty']),
    ExtraTreesClassifier(random_state=RANDOM_STATE, n_estimators=config['best_hparams'][4]['n_estimators'], 
                         min_samples_split=config['best_hparams'][4]['min_samples_split'], 
                         criterion=config['best_hparams'][4]['criterion']),
    MLPClassifier(verbose=False, random_state=RANDOM_STATE, hidden_layer_sizes=config['best_hparams'][5]['hidden_layer_sizes'], 
                  activation=config['best_hparams'][5]['activation'], solver=config['best_hparams'][5]['solver'], 
                  alpha=config['best_hparams'][5]['alpha'])
]
      
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
                
def train_fasttext(input_path):
    """
    Train a FastText model using data from the specified input path.

    Parameters
    ----------
    input_path : str
        Path to the input data file.
    """
    df = pd.read_csv(input_path)
    df = df.dropna(subset=['tweet'])
    X = df['tweet']
    y = df['label']

    try:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_SIZE, random_state=RANDOM_STATE)
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        logger.info("Formatting training set...")
        create_fasttext_format(train_df, XTRAIN_PATH)
        logger.info("Training set formatted.")
        logger.info("Formatting validation set...")
        create_fasttext_format(val_df, XVAL_PATH)
        logger.info("Validation set formatted.")
        logger.info(
            "To start FastText training run:\n"
            ">> src/models/fastText-0.9.2/fasttext supervised -input data/processed/fasttext_train.txt -output models/fasttext_model -autotune-validation data/processed/fasttext_val.txt -verbose 2"
        )
    except Exception as e:
        logger.error(f"Error during FastText training: {e}")
                                 
def train_classifiers(input_path, method, embedding, hparams_tuning, save):
    """
    Train classifiers using the specified method and embedding.

    Parameters
    ----------
    input_path : str
        Path to the input data file.
    method : str
        Training method to use.
    embedding : str
        Embedding method to use.
    hparams_tuning : bool
        Whether to perform hyperparameter tuning.
    save : str
        Whether to train and save the best classifier (based on validation accuracy) or all classifiers.
    """
    logger.info('Loading training data...')
    df = pd.read_csv(input_path)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    df = df.dropna(subset=['tweet'])
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])
    logger.info('Training data loaded.')

    X = df['tweet']
    y = df['label']

    if not hparams_tuning:
        if save == "best":
            logger.info('Creating training set and validation set (holdout)...')
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_SIZE, random_state=RANDOM_STATE)
            if embedding == BOW:
                logger.info('[BoW]: Vectorizing X_train and X_val...')
                vectorizer = CountVectorizer(max_features=5000)
                X_train = vectorizer.fit_transform(X_train)
                X_val = vectorizer.transform(X_val)
                logger.info('[BoW]: X_train and X_val vectorized.')
                logger.info('[BoW]: Saving vectorizer for X_test...')
                joblib.dump(vectorizer, OUTPUT_PATH + "count_vectorizer.pkl")
                logger.info('[BoW]: Vectorizer saved.')
            elif embedding == GLOVE:
                logger.info('[GloVe]: Loading GloVe embeddings...')
                glove_embeddings = load_glove_embeddings(GLOVE_PATH)
                logger.info('[GloVe]: GloVe embeddings loaded.')
                logger.info('[GloVe]: Vectorizing X_train and X_val...')
                X_train = np.array([tweet_to_glove_vector(tweet, glove_embeddings) for tweet in X_train])
                X_val = np.array([tweet_to_glove_vector(tweet, glove_embeddings) for tweet in X_val])
                logger.info('[GloVe]: X_train and X_val vectorized.')   
            else:
                logger.error("'embedding' type is incorrect. Please double check it.")
        
        elif save == "all":
            if embedding == BOW:
                logger.info('[BoW]: Vectorizing X...')
                vectorizer = CountVectorizer(max_features=5000)
                X = vectorizer.fit_transform(X)
                logger.info('[BoW]: X vectorized.')
                logger.info('[BoW]: Saving vectorizer (for further X_test)...')
                joblib.dump(vectorizer, OUTPUT_PATH + "count_vectorizer.pkl")
                logger.info('[BoW]: Vectorizer saved.')
            elif embedding == GLOVE:
                logger.info('[GloVe]: Loading GloVe embeddings...')
                glove_embeddings = load_glove_embeddings(GLOVE_PATH)
                logger.info('[GloVe]: GloVe embeddings loaded.')
                logger.info('[GloVe]: Vectorizing X...')
                X = np.array([tweet_to_glove_vector(tweet, glove_embeddings) for tweet in X])
                logger.info('[GloVe]: X vectorized.')
            else:
                logger.error("'embedding' type is incorrect. Please double check it.")
    else:
        if embedding == BOW:
            logger.info('[BoW]: Vectorizing X...')
            vectorizer = CountVectorizer(max_features=5000)
            X = vectorizer.fit_transform(X)
            logger.info('[BoW]: X vectorized.')
            logger.info('[BoW]: Saving vectorizer (for further X_test)...')
            joblib.dump(vectorizer, OUTPUT_PATH + "count_vectorizer.pkl")
            logger.info('[BoW]: Vectorizer saved.')
        elif embedding == GLOVE:
            logger.info('[GloVe]: Loading GloVe embeddings...')
            glove_embeddings = load_glove_embeddings(GLOVE_PATH)
            logger.info('[GloVe]: GloVe embeddings loaded.')
            logger.info('[GloVe]: Vectorizing X...')
            X = np.array([tweet_to_glove_vector(tweet, glove_embeddings) for tweet in X])
            logger.info('[GloVe]: X vectorized.')
        else:
            logger.error("'embedding' type is incorrect. Please double check it.")

    if hparams_tuning:
        logger.info("Performing hyper-parameters tuning with GridSearch. This will take a long time (few days). It is recommended to run the train.py script with '-hparams_tuning False'.")
        chosen_hparams = list()
        estimators = list()
        results = list()
        for model, model_name, hparams in zip(models, MODEL_NAMES, MODEL_HPARAMS):
            logger.info(f"Training {model_name}...")
            starting_time = time.time()
            clf = GridSearchCV(estimator=model, param_grid=hparams, scoring=SCORING, cv=CV)
            clf.fit(X, y)
            ending_time = time.time()
            chosen_hparams.append(clf.best_params_)
            estimators.append((model_name, clf.best_score_, clf.best_estimator_))
            
            for hparam in hparams:
               logger.info(f"\t--> best value for hyperparameter {hparam}: {clf.best_params_.get(hparam)}")
            
            mean_accuracy = clf.cv_results_['mean_test_score'][clf.best_index_]
            std_score = clf.cv_results_['std_test_score'][clf.best_index_]
            results.append((model_name, model, mean_accuracy, std_score))
            logger.info(f'\t--> best model mean accuracy: {mean_accuracy}')
            logger.info(f'\t--> best model std: {std_score}')
            logger.info(f'\tElapsed time for GridSearch: {timedelta(seconds=ending_time - starting_time)}')
            
        logger.info("Hyper-parameters tuning completed.")
        best_model_name, best_model, best_accuracy, std_score = max(results, key=lambda x: x[2])
        logger.info(f"Best Model: {best_model_name} with accuracy {best_accuracy:.4f}")
    elif save == "best":
        logger.info('Training with holdout validation set...')
        results = []
        for model, model_name in zip(models, MODEL_NAMES):
            logger.info(f"Training {model_name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            results.append((model_name, model, accuracy))
            logger.info(f"Model: {model_name} Accuracy: {accuracy:.4f}")

        logger.info("Training completed.")
        best_model_name, best_model, best_accuracy = max(results, key=lambda x: x[2])
        logger.info(f"Best Model: {best_model_name} with accuracy {best_accuracy:.4f}")

        X = vstack([X_train, X_val])
        y = np.concatenate((y_train, y_val))
        
        logger.info("Final training with best model...")
        best_model.fit(X, y)
        logger.info("Final training completed.")
        logger.info("Saving model...")
        file_path = OUTPUT_PATH + f"{embedding}_{best_model_name}.pkl"
        joblib.dump(best_model, file_path)
        logger.info(f"Model saved at {file_path}")
    
    else:
        for model, model_name in zip(models, MODEL_NAMES):
            logger.info(f"Full training with {model_name}...")
            model.fit(X, y)
            logger.info(f"Full training with {model_name} completed.")
            file_path = OUTPUT_PATH + f"{embedding}_{model_name}.pkl"
            logger.info("Saving model...")
            joblib.dump(model, file_path)
            logger.info(f"Model saved at {file_path}")

def validate_hparams_tuning(ctx, param, value):
    """
    Validate the hparams_tuning option based on the method.

    Parameters
    ----------
    ctx : click.Context
        Click context.
    param : click.Parameter
        Click parameter.
    value : bool
        Value of the hparams_tuning option.

    Returns
    -------
    bool
        Validated value of hparams_tuning.
    """
    method = ctx.params.get('method')
    if method == 'classifiers' and value is None:
        raise click.UsageError("The --hparams_tuning option is required when --method is 'classifiers'")
    if method == 'fastText' and value is None:
        return False  # Default value for fastText
    return value

def train_CNN(input_path):
    """
    Train CNN.

    Parameters
    ----------
    input_path : str
        Path to the input data file.
    
    """
    # 1 - load training data
    logger.info("Loading data...")
    df = pd.read_csv(input_path)
    df = df.dropna(subset=['tweet'])
    X = df['tweet']
    y = df['label']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    # 2 - tokenize and pad sequences
    logger.info("Preprocessing data...")
    max_words = 10000
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    sequence_train = tokenizer.texts_to_sequences(X_train)
    sequence_val = tokenizer.texts_to_sequences(X_val)
    word2vec = tokenizer.word_index
    V = len(word2vec)
    data_train = pad_sequences(sequence_train)
    T = data_train.shape[1]
    data_val = pad_sequences(sequence_val, maxlen=T)
    # 3 - convert into pytorch tensors
    X_train = torch.tensor(data_train, dtype=torch.long)
    y_train = torch.tensor(LabelEncoder().fit_transform(y_train), dtype=torch.float32)
    X_val = torch.tensor(data_val, dtype=torch.long)
    y_val = torch.tensor(LabelEncoder().fit_transform(y_val), dtype=torch.float32)
    # 4 - create dataLoader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
    # 5 - initialize model
    logger.info("Initialize model...")
    model = CNN(vocab_size=V+1, embed_dim=20)
    # 6 - define loss
    criterion = nn.BCEWithLogitsLoss()
    # 7- define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 8 - since overfitting was observed -> early stopping parameters
    wait = 5
    best_val_loss = np.inf
    epochs_no_impr = 0
    early_stop = False
    # 6 - training loop
    logger.info("Start training...")
    epochs = 50
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(texts).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = torch.round(torch.sigmoid(outputs))
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        train_accuracy = 100 * correct_train / total_train
        # 7 - validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for texts, labels in val_loader:
                outputs = model(texts).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = torch.round(torch.sigmoid(outputs))
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = 100 * correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss / len(train_loader)}, Train Accuracy: {train_accuracy}%, '
            f'Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy}%')

        # 8 - early stopping 
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_impr = 0
            logger.info("Saving model...")
            torch.save(model.state_dict(), 'best_new_CNN_model.pt')
        else:
            epochs_no_impr += 1
            if epochs_no_impr >= wait:
                logger.info("early stopping")
                early_stop = True
                break


def train_CNN_LSTM(input_path):
    """
    Train hybrid model CNN-LSTM.

    Parameters
    ----------
    input_path : str
        Path to the input data file.
    
    """
    # 1 - load training data
    logger.info("Loading data...")
    df = pd.read_csv(input_path)
    df = df.dropna(subset=['tweet'])
    X = df['tweet']
    y = df['label']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    # 2 - tokenize and pad sequences
    logger.info("Preprocessing data...")
    max_words = 10000
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    sequence_train = tokenizer.texts_to_sequences(X_train)
    sequence_val = tokenizer.texts_to_sequences(X_val)
    word2vec = tokenizer.word_index
    V = len(word2vec)
    data_train = pad_sequences(sequence_train)
    T = data_train.shape[1]
    data_val = pad_sequences(sequence_val, maxlen=T)
    # 3 - convert into pytorch tensors
    X_train = torch.tensor(data_train, dtype=torch.long)
    y_train = torch.tensor(LabelEncoder().fit_transform(y_train), dtype=torch.float32)
    X_val = torch.tensor(data_val, dtype=torch.long)
    y_val = torch.tensor(LabelEncoder().fit_transform(y_val), dtype=torch.float32)
    # 4 - create dataLoader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
    # 5 - initialize model
    logger.info("Initialize model...")
    model = CNN_LSTM(vocab_size=V+1, embed_dim=20, lstm_hidden_dim=128, num_classes=1)
    # 6 - define loss
    criterion = nn.BCEWithLogitsLoss()
    # 7- define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 8 - since overfitting was observed -> early stopping parameters
    wait = 5
    best_val_loss = np.inf
    epochs_no_impr = 0
    early_stop = False
    # 6 - training loop
    logger.info("Start training...")
    epochs = 50
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(texts).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = torch.round(torch.sigmoid(outputs))
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        train_accuracy = 100 * correct_train / total_train
        # 7 - validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for texts, labels in val_loader:
                outputs = model(texts).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = torch.round(torch.sigmoid(outputs))
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = 100 * correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss / len(train_loader)}, Train Accuracy: {train_accuracy}%, '
            f'Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy}%')

        # 8 - early stopping 
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_impr = 0
            logger.info("Saving model...")
            torch.save(model.state_dict(), 'best_new_CNN_LSTM_model.pt')
        else:
            epochs_no_impr += 1
            if epochs_no_impr >= wait:
                logger.info("early stopping")
                early_stop = True
                break

def train_LSTM_CNN(input_path):
    """
    Train hybrid model LSTM-CNN.

    Parameters
    ----------
    input_path : str
        Path to the input data file.
    
    """
    # 1 - load training data
    logger.info("Loading data...")
    df = pd.read_csv(input_path)
    df = df.dropna(subset=['tweet'])
    X = df['tweet']
    y = df['label']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    # 2 - tokenize and pad sequences
    logger.info("Preprocessing data...")
    max_words = 10000
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    sequence_train = tokenizer.texts_to_sequences(X_train)
    sequence_val = tokenizer.texts_to_sequences(X_val)
    word2vec = tokenizer.word_index
    V = len(word2vec)
    data_train = pad_sequences(sequence_train)
    T = data_train.shape[1]
    data_val = pad_sequences(sequence_val, maxlen=T)
    # 3 - convert into pytorch tensors
    X_train = torch.tensor(data_train, dtype=torch.long)
    y_train = torch.tensor(LabelEncoder().fit_transform(y_train), dtype=torch.float32)
    X_val = torch.tensor(data_val, dtype=torch.long)
    y_val = torch.tensor(LabelEncoder().fit_transform(y_val), dtype=torch.float32)
    # 4 - create dataLoader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
    # 5 - initialize model
    logger.info("Initialize model...")
    model = LSTM_CNN(vocab_size=V+1, embed_dim=20, lstm_hidden_dim=128, num_classes=1)
    # 6 - define loss
    criterion = nn.BCEWithLogitsLoss()
    # 7- define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 8 - since overfitting was observed -> early stopping parameters
    wait = 5
    best_val_loss = np.inf
    epochs_no_impr = 0
    early_stop = False
    # 6 - training loop
    logger.info("Start training...")
    epochs = 50
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(texts).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = torch.round(torch.sigmoid(outputs))
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        train_accuracy = 100 * correct_train / total_train
        # 7 - validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for texts, labels in val_loader:
                outputs = model(texts).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = torch.round(torch.sigmoid(outputs))
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = 100 * correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss / len(train_loader)}, Train Accuracy: {train_accuracy}%, '
            f'Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy}%')

        # 8 - early stopping 
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_impr = 0
            logger.info("Saving model...")
            torch.save(model.state_dict(), 'best_new_LSTM_CNN_model.pt')
        else:
            epochs_no_impr += 1
            if epochs_no_impr >= wait:
                logger.info("early stopping")
                early_stop = True
                break

@click.command()
@click.option('--input', 'input_path', type=str, required=False, help='Path to the training data')
@click.option('--method', type=click.Choice(['classifiers', 'fastText', 'CNN', 'CNN-LSTM','LSTM-CNN','twitter-roberta-base-sentiment-latest','lora-roberta-large-sentiment-latest','bertweet-base','lora-bertweet-large','base-ensemble-random-forest','large-ensemble-random-forest','full-ensemble-random-forest','large-ensemble-nn','extra-trees']), required=True, help='Method to use for training')
@click.option('--embedding', type=click.Choice(['BoW', 'GloVe']), required=False, help='Embedding method to use if method is classifiers')
@click.option('--hparams_tuning', type=bool, callback=validate_hparams_tuning, required=False, help='Whether to use GridSearch K-fold cross-validation for hyper-parameters tuning')
@click.option('--save', type=click.Choice(['all', 'best']), required=False, help='Whether to train and save the best classifier based on validation accuracy or save all classifiers')
@click.option('--validation', type=bool, required=False, help='On LLMs perform training with validation')
def main(input_path, method, embedding, hparams_tuning, save, validation=False):
    """
    Main function to train models based on the specified method, embedding, and hyperparameters tuning.

    Parameters
    ----------
    input_path : str
        Path to the training data.
    method : str
        Method to use for training.
    embedding : str
        Embedding method to use if method is 'classifiers'.
    hparams_tuning : bool
        Whether to perform GridSearch K-fold cross-validation for hyper-parameters tuning.
    """
    if method == 'classifiers' and not embedding:
        raise click.UsageError("Argument --embedding is required when --method is 'classifiers'")
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    if save is None:
        save = "best"
    
    if method == "classifiers":
        train_classifiers(input_path, method, embedding, hparams_tuning, save)
    if method == "fastText":
        train_fasttext(input_path)
    if method =="CNN":
        train_CNN(input_path)
    if method =="CNN-LSTM":
        train_CNN_LSTM(input_path)
    if method =="LSTM-CNN":
        train_LSTM_CNN(input_path)
    if method == "twitter-roberta-base-sentiment-latest":
        fulltwitterrobertabasesentimentlatest.execute(input_path, validation,config)
    if method == "lora-roberta-large-sentiment-latest":
        lora_roberta_large.execute(input_path,validation,config)
    if method == "bertweet-base":
        fullbertweetbase.execute(input_path,validation,config)
    if method == "lora-bertweet-large":
        lora_bertweet_large.execute(input_path,validation,config)
    if method == "base-ensemble-random-forest":
        ensembles.random_forest(["data/embeddings/finetuned-bertweet-base_train_small.pt", "data/embeddings/finetuned-twitter-roberta-base-sentiment-latest_train_small.pt"],"data/processed/train_small.csv",["data/embeddings/finetuned-bertweet-base_test.pt","data/embeddings/finetuned-twitter-roberta-base-sentiment-latest_test.pt"])
    if method == "large-ensemble-random-forest":
        ensembles.random_forest(["data/embeddings/lora-bertweet-large_train_small.pt", "data/embeddings/lora-twitter-roberta-large-topic-sentiment-latest_train_small.pt"],"data/processed/train_small.csv",["data/embeddings/lora-bertweet-large_test.pt","data/embeddings/lora-twitter-roberta-large-topic-sentiment-latest_test.pt"])
    if method == "full-ensemble-random-forest":
        ensembles.random_forest(["data/embeddings/lora-bertweet-large_train_small.pt", "data/embeddings/lora-twitter-roberta-large-topic-sentiment-latest_train_small.pt","data/embeddings/finetuned-bertweet-base_train_small.pt","data/embeddings/finetuned-twitter-roberta-base-sentiment-latest_train_small.pt"],"data/processed/train_small.csv",["data/embeddings/lora-bertweet-large_test.pt","data/embeddings/lora-twitter-roberta-large-topic-sentiment-latest_test.pt","data/embeddings/finetuned-bertweet-base_test.pt","data/embeddings/finetuned-twitter-roberta-base-sentiment-latest_test.pt"])
    if method == "large-ensemble-nn":
        ensembles.ensemble(["data/embeddings/lora-bertweet-large_train_small.pt", "data/embeddings/lora-twitter-roberta-large-topic-sentiment-latest_train_small.pt"],"data/processed/train_small.csv",["data/embeddings/lora-bertweet-large_test.pt","data/embeddings/lora-twitter-roberta-large-topic-sentiment-latest_test.pt"], config, validation=validation)
    if method == "extra-trees":
        ensembles.extra_trees(["data/embeddings/lora-bertweet-large_train_small.pt", "data/embeddings/lora-twitter-roberta-large-topic-sentiment-latest_train_small.pt"],"data/processed/train_small.csv",["data/embeddings/lora-bertweet-large_test.pt","data/embeddings/lora-twitter-roberta-large-topic-sentiment-latest_test.pt"])
if __name__ == "__main__":
    main()
