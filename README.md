# ETH Computational Intelligence Lab Project 2024
## Twitter Text Sentiment Classification

- Group Name: CIL'em All
- Authors: Christopher Zanoli, Federica Bruni, Francesco Rita, Matteo Boglioni

## 1. Project Description
The goal of this project is to ... //TODO

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
Install project's dependencies by running:
```
make requirements
```
Preprocess the raw training data by running:
```
python src/data/preprocess_train.py
```
The pre-processed training data will be saved in ```data/processed```

## 4. Project Results

| Model                              | Accuracy(\%) | Variance(\%) |
|------------------------------------|--------------|--------------|
| Example1                           | 80.23        | 0.208        |
| Example 2                          | **92.04**    | 0.207        |

Final remarks ...

