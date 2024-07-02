# ETH Computational Intelligence Lab Project 2024
## Twitter Text Sentiment Classification

- Group Name: CIL'em All
- Authors: Christopher Zanoli, Federica Bruni, Francesco Rita, Matteo Boglioni

## 1. Project Description
The goal of the project is to classify if the sentiment of Twitter tweets is positive or negative.

## 2. Project Organization
    |
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    |   ├── interim        <- Generated code results to be used in reporting
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts and config to download or generate data
        |   ├── generate_dataset.py
        |   ├── config.yml
        |   ├── vocabulary.py
        │   └── preprocess_train.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        ├── models         <- Scripts and config to train models and then use trained models to make
        │   │                 predictions
        │   ├── download_glove_fasttext.py
        │   ├── config.yml
        │   ├── predict.py
        │   └── train.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py

## 3. Project Setup
Create a virtual environment and install project's dependencies by running:
```
pip install -r requirements.txt
```
In the project ```root``` folder create a folder ```data``` and two subfolders: ```raw``` and ```processed```.
Download into the raw subfolder the data from [here](https://www.kaggle.com/competitions/ethz-cil-text-classification-2024/data):

```
root
  └── data
        ├── processed
        └── raw
             ├── test_data.txt
             ├── train_neg_full.txt
             ├── train_neg.txt
             ├── train_pos_full.txt
             └── train_pos.txt
```

## 4. Project Pipeline
From the project ```root``` folder, preprocess the raw training and test data by running:
```
python src/data/preprocess_data.py
```
The pre-processed training and test data will be saved in ```data/processed``` as ```train.csv``` and ```test.csv```

From the project ```root``` folder, download the GloVe embeddings and the FastText source code  by running:
```
python src/models/download_glove_fasttext.py
```
This will automatically download and unzip the ```glove.twitter.27B.zip``` file containing the GloVe embeddings and the ```fastText-0.9.2.zip``` file containing the source code for the FastText pipeline. Both will be saved at ```src/models```
Now run the following commands to generate FastText binaries:
```
cd src/models/fastText-0.9.2
make
```

### Embedding (BoW, GloVe) + Classifiers training Pipeline
- To run the BoW + Classifiers pipeline run (from ```root``` folder of the project):
    ```
    python src/models/train.py --input data/processed/train.csv --method classifiers --embedding BoW --hparams_tuning False
    ```
- To run the BoW + Classifiers pipele with hyper-parameterstuning through GirdSearchCV run (from ```root``` folder of the project):
    ```
    python src/models/train.py --input data/processed/train.csv --method classifiers --embedding BoW --hparams_tuning True
    ```
    ⚠️ Please note that hyper-parameters tuning requires a few days to be completed.

- To run the GloVe + Classifiers pipeline run (from ```root``` folder of the project):
    ```
    python src/models/train.py --input data/processed/train.csv --method classifiers --embedding BoW --hparams_tuning False
    ```
- To run the GloVe + Classifiers pipeline with hyper-parameterstuning through GirdSearchCV run (from ```root``` folder of the project):
    ```
    python src/models/train.py --input data/processed/train.csv --method classifiers --embedding BoW --hparams_tuning True
    ```
    ⚠️ Please note that hyper-parameters tuning requires a few days to be completed.

    ⚠️ We observed that, depending on the platform running the code, the process of the pipeline using GloVe embeddings might be internally killed by the system due to very high memory usage. If you experience such issue, please consider using the small version of the training set by setting the ```train_dataset_type``` flag in ```src/data/config.yml``` file to ```small``` or use GloVe embeddings with fewer dimensions, for example 25, by setting the flag ```GLOVE_VECTOR_SIZE``` to ```25``` and glove_path to ```src/models/glove.twitter.27B/glove.twitter.27B.25d.txt``` in ```src/models/config.yml```.

The final best model will be saved in the ```models``` folder of the ```root``` of the project.

### FastText Training Pipeline
From ```root``` folder of the project run:
```
python src/models/train.py --input data/processed/train.csv --method fastText
```
Then run:
```
src/models/fastText-0.9.2/fasttext supervised -input data/processed/fasttext_train.txt -output models/fasttext_model -autotune-validation data/processed/fasttext_val.txt -verbose 2
```
This will start the training with hyper-parameters tuning and it will last 5 minutes. Then it will perform the final training with the best set of hyperparameters-found. The validation set accuracy is displayed next to "Best Score:".

### CNN Training Pipeline
//TODO

### RNN Training Pipeline
//TODO

### LLM Training Pipeline
//TODO

### Making Predictions:
- Embedding (BoW, GloVe) + Classifiers Pipeline (from root folder of the project):
```
python src/models/predict.py --model models/classifiers.pkl --data data/processed/test.csv --method classifiers --embedding BoW
```
- FastText (from root folder of the project):
```
python src/models/predict.py --model models/fasttext_model.bin --data data/processed/test.csv --method fastText
```
- CNN:
```
//TODO
```
- RNN:
```
//TODO
```
- LLM:
```
//TODO
```


## 4. Project Results
The following table shows the obtained final mean accuracy and standard deviation of the ```best model``` for each pipeline using ```K-fold cross validation``` (k=5) through ```GridSearchCV```.

| Model                              | Accuracy (\%) | Std (\%) |
|------------------------------------|---------------|----------|
| BoW + Logistic Regressor           | 80.60         |   0.05   |
| BoW + Support Vector Machine       | 80.59         |   0.18   |
| BoW + Ridge Classifier             | 80.46         |   0.17   |
| BoW + SGD Classifier               | 80.18         |   0.20   |
| BoW + Extra Trees                  | 81.58         |   0.27   |
| BoW + Multi Layer Perceptron       | 79.56         |   0.31   |
| GloVe + Logistic Regressor         | 78.70         |   0.26   |
| GloVe + Support Vector Machine     | 78.96         |   0.26   |
| GloVe + Ridge Classifier           | 78.68         |   0.26   |
| GloVe + SGD Classifier             | 78.20         |   0.23   |
| GloVe + Extra Trees                | 78.58         |   0.30   |
| GloVe + Multi Layer Perceptron     | 81.02         |   0.36   |
| FastText Classifier                | **86.21**     |   0.13   |


Final remarks:
- //TODO: cpu specs for training sklearn classifiers and fasttext
- //TODO_ gpu specs for training of CNN, RNN and LLM

