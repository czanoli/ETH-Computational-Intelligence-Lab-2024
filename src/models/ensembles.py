import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import ExtraTreesClassifier
import torch
from train_llms import train
from utils import *



def random_forest(embedding_paths, labels_path, test_paths,seed=42, validation=False):
    """
    Train a Random Forest classifier on (potentially) multiple embeddings concatenated and make predictions on test data.

    Parameters
    ----------
    embedding_paths : list[str]
        List of paths to embeddings datasets.
    labels_path : str
        Path to the original dataset with labels.
    test_paths : list[str]
        List of paths to test embeddings datasets.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    validation: bool, otpional
        Flag to indicate whether to use a validation set. Default is False.

    Returns
    -------
    None
        Saves test predictions in predictions.csv
    """

    set_seed(seed)
    X_train, y_train = couple_data(embedding_paths, labels_path)
    if validation:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=seed, shuffle=True)
    print("Data coupled")

    clf = RandomForestClassifier(n_estimators=100, random_state=seed, verbose=2)
    clf.fit(X_train, y_train)
    print("Model fit")
    if validation:
        y_val_pred = clf.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        print("Validation accuracy ", val_accuracy)

    X_test = couple_data(test_paths)
    y_pred = clf.predict(X_test)
    y_pred = [-1 if y==0 else 1 for y in y_pred]

    save_predictions(y_pred)


def ensemble(embeddings_paths, labels_path, test_paths, configfile, seed= 42, validation=False):
    """
    Train an ensemble model on embeddings and make predictions on test data.

    Parameters
    ----------
    embeddings_paths : list[str]
        List of paths to embeddings datasets.
    labels_path : str
        Path to the original dataset with labels.
    test_paths : list[str]
        List of paths to test embeddings datasets.
    configfile : dict
        Configuration parameters loaded from a .yml file.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    validation : bool, optional
        Flag to indicate whether to use a validation set. Default is False.

    Returns
    -------
    None
        Saves test predictions in predictions.csv
    """
    set_seed(seed)
    hidden_size = 0
    for path in embeddings_paths:
        hidden_size += len(np.array(torch.load(path))[0])
    model = Ensemble(hidden_size)
    train_loader, val_loader = get_embeddings_loader(embeddings_paths, labels_path= labels_path, seed= seed, validation= validation)
    model =  train(train_loader, model, lr= configfile['ensemble']['lr'], num_epochs= configfile['ensemble']['epochs'], val_loader= val_loader)
    test_loader = get_embeddings_loader(test_paths)
    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with torch.no_grad():
        predictions = []
        for batch in test_loader:
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
