# ETH Computational Intelligence Lab Project 2024
## Twitter Text Sentiment Classification

- Group Name: CIL'em All
- Authors: Christopher Zanoli, Federica Bruni, Francesco Rita, Matteo Boglioni

## 1. Project Description
The goal of the project is to ... //TODO

## 2. Project Organization
    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
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
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   ├── predict_model.py
        │   └── train_model.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py

## 3. Project Setup
We will be using ```make``` as a task runner (i.e. for setting up the virtual environment, creating the pre-processed dataset, initiating the training, etc...). [GNU Make](https://www.gnu.org/software/make/) is typically pre-installed on Linux and macOS systems. If you are using Windows, you may need to install Make. See the [Installing Make on Windows](https://cookiecutter-data-science.drivendata.org/using-the-template/#installing-make-on-windows) for further information.


## 4. Project Pipeline
Create a virtual environment and install project's dependencies by running:
```
pip install -r requirements.txt
```
From the project root folder, preprocess the raw training and test data by running:
```
python src/data/preprocess_data.py
```
The pre-processed training and test data will be saved in ```data/processed``` as ```train.csv``` and ```test.csv```

From the project root folder, download the GloVe embeddings and the FastText source code  by running:
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
- To run the BoW + Classifiers pipeline run (from root folder of the project):
    ```
    python src/models/train.py --input data/processed/train.csv --method classifiers --embedding BoW --hparams_tuning False
    ```
- To run the BoW + Classifiers pipele with hyper-parameterstuning through GirdSearchCV run (from root folder of the project):
    ```
    python src/models/train.py --input data/processed/train.csv --method classifiers --embedding BoW --hparams_tuning True
    ```
    ⚠️ Please note that hyper-parameters tuning requires a few days to be completed.


- To run the GloVe + Classifiers pipeline run (from root folder of the project):
    ```
    python src/models/train.py --input data/processed/train.csv --method classifiers --embedding BoW --hparams_tuning False
    ```
- To run the GloVe + Classifiers pipeline with hyper-parameterstuning through GirdSearchCV run (from root folder of the project):
    ```
    python src/models/train.py --input data/processed/train.csv --method classifiers --embedding BoW --hparams_tuning True
    ```
    ⚠️ Please note that hyper-parameters tuning requires a few days to be completed.

The final best model will be saved in the ```models``` folder of the root of the project.

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


To make pedictions run:
```
./fasttext predict src/models/fasttext_model data/processed/test.txt > results/predictions.txt
```

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
src/models/fastText-0.9.2/fasttext predict models/fasttext_model data/processed/fasttext_test.txt > results/fasttext_predictions.txt
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
The following table shows the obtained final accuracies on the validation set for each pipeline.

| Model                              | Accuracy(\%) |
|------------------------------------|--------------|
| Example1                           | 80.23        |
| Example 2                          | **92.04**    |

Final remarks ...

