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

warnings.filterwarnings("ignore", category=ConvergenceWarning)
logger = logging.getLogger(__name__)

# Load config
with open(Path(__file__).resolve().parent/'config.yml', 'r') as ymlfile:
    config = yaml.safe_load(ymlfile)
    
for hps in config['models_hparams_']:
    if 'hidden_layer_sizes' in hps:
        hps['hidden_layer_sizes'] = [tuple(size) for size in hps['hidden_layer_sizes']]

RANDOM_STATE = config['random_state']
GLOVE_PATH = config['glove_path']
BOW = config['BOW']
GLOVE = config['GLOVE']
VAL_SIZE = config['VAL_SIZE']
CV = config['CV']
SCORING = config['SCORING']
models = [
    LogisticRegression(random_state=RANDOM_STATE),
    LinearSVC(random_state=RANDOM_STATE),
    RidgeClassifier(random_state=RANDOM_STATE),
    SGDClassifier(random_state=RANDOM_STATE),
    ExtraTreesClassifier(random_state=RANDOM_STATE),
    MLPClassifier(verbose=False, random_state=RANDOM_STATE),
]
MODEL_NAMES = config['models_names']
MODEL_HPARAMS = config['models_hparams']
OUTPUT_PATH = config['OUTPUT_PATH']
XTRAIN_PATH = config['fasttext']['xtrain_path']
XVAL_PATH = config['fasttext']['xval_path']
X_PATH = config['fasttext']['xfull_path']
FASTTEXT_HPARAMS_PATH = config['fasttext']['hparams_path']
FASTTEXT_PATH = config['fasttext']['folder_path']
GLOVE_VECTOR_SIZE = config['GLOVE_VECTOR_SIZE']

            
def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def tweet_to_glove_vector(tweet, embeddings, vector_size=GLOVE_VECTOR_SIZE):
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

def save_best_hyperparameters(output):
    best_params = {}
    for line in output.split('\n'):
        if line.startswith('-'):
            key, value = line[1:].split(': ')
            best_params[key] = value
    with open(FASTTEXT_HPARAMS_PATH, 'w') as file:
        yaml.dump(best_params, file)

def load_hyperparameters():
    with open(FASTTEXT_HPARAMS_PATH, 'r') as file:
        return yaml.safe_load(file)

def create_fasttext_format(df, file_path, is_test=False):
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
                
                
        #fasttext_command = f"{FASTTEXT_PATH}/fasttext supervised -input {XTRAIN_PATH} -output fasttext_model -autotune-validation {XVAL_PATH} -verbose 2"
        '''
        else:
            logger.info("Formatting training data ...")
            create_fasttext_format(df, X_PATH)
            logger.info("Training data formatted.")
            logger.info("Loading fastText hyper-parameters...")
            best_params = load_hyperparameters()
            logger.info("FastText hyper-parameters loaded.")
            params = " ".join([f"-{key} {value}" for key, value in best_params.items()])
            fasttext_command = f"{FASTTEXT_PATH}/fasttext supervised -input {X_PATH} -output fasttext_model {params} -verbose 2"
        '''
    except Exception as e:
        logger.error(f"Error during FastText training: {e}")

def batch_generator(file_path, batch_size):
    for chunk in pd.read_csv(file_path, chunksize=batch_size):
        chunk = chunk.dropna(subset=['tweet'])
        yield chunk
        
def process_batches(file_path, glove_embeddings, batch_size=20000):
    X = []
    y = []
    for chunk in batch_generator(file_path, batch_size):
        tweets = chunk['tweet'].tolist()
        labels = chunk['label'].tolist()
        
        X.extend([tweet_to_glove_vector(tweet, glove_embeddings) for tweet in tweets])
        y.extend(labels)
        
    return np.array(X), np.array(y)
                                 
def train_classifiers(input_path, method, embedding, hparams_tuning):
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
        if embedding == BOW:
            logger.info('[BoW]: Vectorizing X...')
            vectorizer = CountVectorizer(max_features=5000)
            X = vectorizer.fit_transform(X)
            logger.info('[BoW]: X vectorized.')
            logger.info('[BoW]: Saving vectorizer for X_test...')
            joblib.dump(vectorizer, OUTPUT_PATH + "count_vectorizer.pkl")
            logger.info('[BoW]: Vectorizer saved.')
        elif embedding == GLOVE:
            logger.info('[GloVe]: Loading GloVe embeddings...')
            glove_embeddings = load_glove_embeddings(GLOVE_PATH)
            logger.info('[GloVe]: GloVe embeddings loaded.')
            logger.info('[GloVe]: Vectorizing X...')
            X = np.array([tweet_to_glove_vector(tweet, glove_embeddings) for tweet in X])
            #logger.info('[GloVe]: Vectorizing data in batches...')
            #X, y = process_batches(input_path, glove_embeddings)
            #logger.info('[GloVe]: Data vectorized.')
            logger.info('[GloVe]: X vectorized.')

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
            
            # Save models with repsective accuracy
            results.append((model_name, model, mean_accuracy, std_score))

            logger.info(f'\t--> best model mean accuracy: {mean_accuracy}')
            logger.info(f'\t--> best model std: {std_score}')
            logger.info(f'\tElapsed time for GridSearch: {timedelta(seconds=ending_time - starting_time)}')
            
        logger.info("Hyper-parameters tuning completed.")
        best_model_name, best_model, best_accuracy, std_score = max(results, key=lambda x: x[2])
        logger.info(f"Best Model: {best_model_name} with accuracy {best_accuracy:.4f}")
    else:
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
    file_path = OUTPUT_PATH + f"{method}.pkl"
    joblib.dump(best_model, file_path)
    logger.info(f"Model saved at {file_path}")


def validate_hparams_tuning(ctx, param, value):
    method = ctx.params.get('method')
    if method == 'classifiers' and value is None:
        raise click.UsageError("The --hparams_tuning option is required when --method is 'classifiers'")
    if method == 'fastText' and value is None:
        return False  # Default value for fastText
    return value


@click.command()
@click.option('--input', 'input_path', type=str, required=True, help='Path to the training data')
@click.option('--method', type=click.Choice(['classifiers', 'fastText', 'CNN', 'RNN']), required=True, help='Method to use for training')
@click.option('--embedding', type=click.Choice(['BoW', 'GloVe']), required=False, help='Embedding method to use if method is classifiers')
@click.option('--hparams_tuning', type=bool, callback=validate_hparams_tuning, required=False, help='Whether to use GridSearch K-fold cross-validation for hyper-parameters tuning')
def main(input_path, method, embedding, hparams_tuning):
    if method == 'classifiers' and not embedding:
        raise click.UsageError("Argument --embedding is required when --method is 'classifiers'")
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    if method == "classifiers":
        train_classifiers(input_path, method, embedding, hparams_tuning)
    if method == "fastText":
        train_fasttext(input_path)

if __name__ == "__main__":
    main()
