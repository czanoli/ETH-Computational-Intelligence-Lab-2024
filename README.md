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
- To run the CNN pipeline run (from ```root``` folder of the project):
    ```
    python src/models/train.py --input data/processed/train.csv --method CNN
    ```

### HYBRID MODELS Training Pipeline
- To run the CNN-LSTM pipeline run (from ```root``` folder of the project):
    ```
    python src/models/train.py --input data/processed/train.csv --method CNN-LSTM
    ```
- To run the LSTM-CNN pipeline run (from ```root``` folder of the project):
    ```
    python src/models/train.py --input data/processed/train.csv --method LSTM_CNN
    ```

### LLM Training Pipeline
- To run the twitter-roberta-base-sentiment-latest pipeline run (from ```root``` folder of the project):
    ```
    python src/models/train.py --input data/processed_llm/train_full.csv --method twitter-roberta-base-sentiment-latest
    ```
- To run the lora-roberta-large-sentiment-latest pipeline run (from ```root``` folder of the project):
    ```
    python src/models/train.py --input data/processed_llm/train_full.csv --method lora-roberta-large-sentiment-latest
    ```
- To run the bertweet-base pipeline run (from ```root``` folder of the project):
    ```
    python src/models/train.py --input data/processed_llm/train_full.csv --method bertweet-base
    ```
- To run the lora-bertweet-large pipeline run (from ```root``` folder of the project):
    ```
    python src/models/train.py --input data/processed_llm/train_full.csv --method lora-bertweet-large
    ```
At the end of the training pipeline the finetuned model is saved in ```models``` folder.

### ENSEMBLES Training Pipeline 
In order to perform the ENSEMBLE & BERT-EFRI Training Pipeline it's firstly needed to generate the embeddings dataset with the following commands (from ```root``` folder of the project):
    
    ```
    python src/models/generate_embeddings.py --model_path path_to_finetuned_model --data_path data/processed_llm/train_small.csv --model_name name_of_model
    ```
    ```
    python src/models/generate_embeddings.py --model_path path_to_finetuned_mode --data_path data/processed_llm/test.csv --model_name name_of_model
    ```
Where the name_of_model is the name of the specific model in the ```/src/models/config.yml``` file and path_to_finetuned_model is the path the finetuned model' weights are saved in. For example, to obtain bertweet-base embeddings run (from ```root``` folder of the project):
    ```
    python src/models/generate_embeddings.py --model_path models/finetuned-bertweet-base  --data_path data/processed_llm/train_small.csv --model_name models_bertweet_base
    ```
    ```
    python src/models/generate_embeddings.py --model_path models/finetuned-bertweet-base --data_path data/processed_llm/test.csv --model_name models_bertweet_base
    ```
- To run the base-ensemble-nn fed with the embeddings of the base-models pipeline run (from ```root``` folder of the project):
    ```
    python src/models/train.py --method base-ensemble-nn
    ```
- To run the large-ensemble-nn fed with the embeddings of the large-models pipeline run (from ```root``` folder of the project):
    ```
    python src/models/train.py --method large-ensemble-nn
    ```
- To run the full-ensemble-nn fed with the embeddings of all the llm models pipeline run (from ```root``` folder of the project):
    ```
    python src/models/train.py --method full-ensemble-nn
    ```
### BERT-EFRI Training Pipeline 
- To run the base-ensemble-random-forest fed with the embeddings of the base-models pipeline run (from ```root``` folder of the project):
    ```
    python src/models/train.py --method base-ensemble-random-forest
    ```
- To run the large-ensemble-random-forest fed with the embeddings of the large-models pipeline run (from ```root``` folder of the project):
    ```
    python src/models/train.py --method large-ensemble-random-forest
    ```
- To run the full-ensemble-random-forest fed with the embeddings of all the llm models pipeline run (from ```root``` folder of the project):
    ```
    python src/models/train.py --method full-ensemble-random-forest
    ```


### Making Predictions:
- Embedding (BoW, GloVe) + Classifiers Pipeline (from ```root``` folder of the project):
```
python src/models/predict.py --model models/<your-classifier>.pkl --data data/processed/test.csv --method classifiers --embedding BoW
```
Note: replace ```<your-classifier>.pkl``` with the name of the classifier you generated during training. For example: ```BoW_Logistic_Regressor.pkl```
- FastText (from ```root``` folder of the project):
```
python src/models/predict.py --model models/fasttext_model.bin --data data/processed/test.csv --method fastText
```
- CNN (from ```root``` folder of the project):
```
python src/models/predict.py --data data/processed/test.csv --method CNN
```
- HYBRID MODELS (from ```root``` folder of the project):
```
python src/models/predict.py --data data/processed/test.csv --method CNN-LSTM
```
```
python src/models/predict.py --data data/processed/test.csv --method LSTM-CNN
```
- LLM (from ```root``` folder of the project):
```
python src/models/predict.py --data data/processed_llm/test.csv --method twitter-roberta-base-sentiment-latest
```
```
python src/models/predict.py --data data/processed_llm/test.csv --method lora-roberta-large-sentiment-latest
```
```
python src/models/predict.py --data data/processed_llm/test.csv --method bertweet-base
```
```
python src/models/predict.py --data data/processed_llm/test.csv --method lora-bertweet-large
```
- ENSEMBLES & BERT-EFRI :
The training pipeline of each configuration compute the predictions and save them in the ```root``` folder as predictions.csv


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
  
    | Model                     | Accuracy (\%) |
    |---------------------------|---------------|
    | CNN(v1)                   | 80.32         |
    | CNN(v2) final choice      | 84.14         |
    | CNN(v3)                   | 83.78         |


- Hybrid Models with CNN-LSTM and LSTM-CNN Architectures:
  
    | Model                     | Accuracy (\%) |
    |---------------------------|---------------|
    | CNN-LSTM(v1)              | 84.55         |
    | CNN-LSTM(v2) final choice | 85.72         |
    | LSTM-CNN(v1)              | 86.93         |
    | LSTM-CNN(v2) final choice | 87.12         |

- Large Language Models:

    | Model                     | Accuracy (\%) |
    |---------------------------|---------------|
    | RoBERTa-base              | 90.88         |
    | RoBERTa-large             | 91.66         |
    | BERTweet-base             | 91.20         |
    | BERTweet-large            | 91.64         |

- Ensembles:

    | Model                     | Accuracy (\%) |
    |---------------------------|---------------|
    | Ensemble-base             | 91.60         |
    | Ensemble-large            | 91.66         |
    | Ensemble-base&large       | 90.60         |

- Novel Solution:

    | Model                     | Accuracy (\%) |
    |---------------------------|---------------|
    | BERT-EFRI-base            | 91.64         |
    | BERT-EFRI-large           | 91.84         |
    | BERT-EFRI-base&large      | 91.14         |

## 5. Remarks

- [scikit-learn](https://scikit-learn.org/stable/index.html) classifiers and [FastText](https://fasttext.cc/) classifier have been trained on the following CPU: Intel(R) Core(TM) i7-8565U CPU @ 1.80GHz
- LLMs have been trained over NVIDIA GPUs with **minimum** 11GB VRAM, CUDA Version 12.6

