import pandas as pd
import yaml
from pathlib import Path

# Load configuration from config.yaml
with open(Path(__file__).resolve().parent/'config.yml', 'r') as file:
    config = yaml.safe_load(file)
train_files = config['raw_train_paths']
test_file = config['raw_test_path']

class DataProcessor:
    def __init__(self, train_files, test_file, output_file):
        self.train_files = train_files
        self.test_file = test_file
        self.output_file = output_file
    
    def load_training_data(self, dataset_type):
        if dataset_type not in self.train_files.keys():
            raise ValueError(f"Invalid training dataset_type. Expected one of {list(train_files.keys())}")
        df = pd.DataFrame()
        for file in self.train_files[dataset_type]:
            label = "positive" if "pos" in file else "negative" if "neg" in file else None
            with open(file, 'r', encoding='utf-8') as file:
                tweets = file.readlines()
            tmp_df = pd.DataFrame(tweets, columns=["tweet"])
            tmp_df["label"] = label
            df = pd.concat([df, tmp_df], ignore_index=True)
        return df

    def load_test_data(self, dataset_type="test"):
        if dataset_type not in test_file.keys():
            raise ValueError(f"Invalid test dataset_type. Expected {list(train_files.keys())}")
        df = pd.DataFrame()
        with open(self.test_file[dataset_type], 'r', encoding='utf-8') as file:
            data = [line.split(',', 1) for line in file.readlines()]
        df = pd.DataFrame(data, columns=["id", "tweet"])
        return df
    
    def save_df_to_csv(self, df):
        df.to_csv(self.output_file, index=False)

