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
Note: by default, the ```full``` version of the dataset is used. The pre-processed training and test data will be saved in ```data/processed``` as ```train.csv``` and ```test.csv```

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

### Embedding + Classifiers training Pipeline (BoW, GloVe, FastText)
- To run the BoW + Classifiers pipeline run (from ```root``` folder of the project):
    ```
    python src/models/train.py --input data/processed/train.csv --method classifiers --embedding BoW --hparams_tuning False --save all
    ```
    Note: models are instantiated with the best hyper-parameters found through ```K-fold cross validation``` (```GridSearchCV```, k=5). You can set the ```--save``` flag to ```best``` to save only the best model (based on validation accuracy).
- To run the BoW + Classifiers pipele with hyper-parameterstuning through GirdSearchCV run (from ```root``` folder of the project):
    ```
    python src/models/train.py --input data/processed/train.csv --method classifiers --embedding BoW --hparams_tuning True 
    ```
    ⚠️ Please note that hyper-parameters tuning requires a few days to be completed.

- To run the GloVe + Classifiers pipeline run (from ```root``` folder of the project):
    ```
    python src/models/train.py --input data/processed/train.csv --method classifiers --embedding BoW --hparams_tuning False --save all
    ```
    Note: models are instantiated with the best hyper-parameters found through ```K-fold cross validation``` (```GridSearchCV```, k=5). You can set the ```--save``` flag to ```best``` to save only the best model (based on validation accuracy).
- To run the GloVe + Classifiers pipeline with hyper-parameterstuning through GirdSearchCV run (from ```root``` folder of the project):
    ```
    python src/models/train.py --input data/processed/train.csv --method classifiers --embedding BoW --hparams_tuning True
    ```
    ⚠️ Please note that hyper-parameters tuning requires a few days to be completed.

⚠️ We observed that, depending on the platform executing the code, the system might terminate some pipelines due to excessive memory usage (MemoryError). If you encounter this issue, consider increasing your system's available memory or using the smaller dataset version by setting  ```train_dataset_type: "small" ``` in ```src/data/config.yml```. Specifically for the GloVe pipeline, try running it from the Jupyter notebook named ```1.0--cz-BoW-GloVe.ipynb``` in the ```notebooks``` folder. If this does not resolve the issue, either use the smaller version of the training set as previously mentioned or use GloVe embeddings with fewer dimensions, such as 25, by setting the ```GLOVE_VECTOR_SIZE``` flag to ```25``` and updating the ```glove_path``` to ```src/models/glove.twitter.27B/glove.twitter.27B.25d.txt``` in ```src/models/config.yml```.

The final best model will be saved in the ```models``` folder of the ```root``` of the project.

- To run the FastText pipeline run (from ```root``` folder of the project):
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
The table below presents the test accuracies achieved by the selected baselines and the novel solution, evaluated on 50% of the public test data.

- Word Embeddings with Classifiers:

    | Model                     | Accuracy (\%) |
    |---------------------------|---------------|
    | BoW + Logistic Regressor  | 80.04         |
    | BoW + SVM                 | 79.74         |
    | BoW + Ridge Classifier    | 79.58         |
    | BoW + SGD Classifier      | 79.08         |
    | BoW + Extra Trees         | 79.74         |
    | BoW + MLP                 | 83.30         |
    | GloVe + Logistic Regressor| 77.70         |
    | GloVe + SVM               | 77.82         |
    | GloVe + Ridge Classifier  | 78.04         |
    | GloVe + SGD Classifier    | 77.98         |
    | GloVe + Extra Trees       | 78.66         |
    | GloVe + MLP               | 83.90         |
    | FastText                  | 85.88         |

- Convolutional Neural Networks (CNN):
  
    //TODO

- Hybrid Models with CNN-LSTM and LSTM-CNN Architectures:
  
    //TODO

- Large Language Models:

    //TODO

- Novel Solution:

    //TODO


## 5. Remarks
//TODO

- [scikit-learn](https://scikit-learn.org/stable/index.html) classifiers and [FastText](https://fasttext.cc/) classifier have been trained on the following CPU: Intel(R) Core(TM) i7-8565U CPU @ 1.80GHz
- //TODO_ gpu specs for training of CNN, RNN and LLM

