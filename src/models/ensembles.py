import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import torch
from utils import *



def random_forest(embedding_paths, labels_path, test_paths,seed=42):
    set_seed(seed)
    x, y = couple_data(embedding_paths, labels_path)
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=seed, shuffle=True)
    print("Data coupled")
    clf = RandomForestClassifier(n_estimators=100, random_state=seed, verbose=2)
    clf.fit(X_train, y_train)
    print("Model fit")
    y_val_pred = clf.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print("Validation accuracy ", val_accuracy)
    X_test = couple_data(test_paths)
    y_pred = clf.predict(X_test)
    save_predictions(y_pred)



