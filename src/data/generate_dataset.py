import pandas as pd
import yaml
from pathlib import Path
from ydata_profiling import ProfileReport
import re
import contractions
import emoji

from vocabulary import *

# Load configuration from config.yaml
with open(Path(__file__).resolve().parent/'config.yml', 'r') as file:
    config = yaml.safe_load(file)
train_files = config['raw_train_paths']
test_file = config['raw_test_path']

class DataProcessor:
    def __init__(self, nan_policy, duplicates_policy, shared_duplicates_policy, conflict_policy, 
                 dataset_type, prj_dir, train_files, test_file, output_file):
        self.nan_policy = nan_policy
        self.duplicates_policy = duplicates_policy
        self.shared_duplicates_policy = shared_duplicates_policy
        self.conflict_policy = conflict_policy
        self.dataset_type = dataset_type
        self.prj_dir = prj_dir
        self.train_files = train_files
        self.test_file = test_file
        self.output_file = output_file
    
    def load_training_data(self):
        if self.dataset_type not in self.train_files.keys():
            raise ValueError(f"Invalid training dataset_type. Expected one of {list(self.train_files.keys())}")
        df = pd.DataFrame()
        for file in self.train_files[self.dataset_type]:
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
        
    def nulls_info(self, df):
        return df.isnull().sum()
    
    def profile(self, df):
        profile = ProfileReport(df, title="Twitter Sentiment EDA Report", minimal=True)
        profile.to_file(self.prj_dir / f"reports/twitter_sentiment_eda_{self.dataset_type}.html")
    
    def process_dataframe(self, df, df2=None):
        """
        Process the DataFrame based on the specified policies.
        
        Args:
        df (pd.DataFrame): The input DataFrame.
        df2 (pd.DataFrame, optional): The second DataFrame for shared duplicates processing. Default is None.
        
        Returns:
        tuple: The processed DataFrame(s). If df2 is None, returns (df,). Otherwise, returns (df, df2).
        """
        # 1. Handle null values
        if self.nan_policy == "drop":
            df = df.dropna()
        
        # 2. Handle duplicates
        if self.duplicates_policy == "drop":
            df = df.drop_duplicates()
        elif self.duplicates_policy == "keep":
            df = df[df.duplicated(keep=False)]
            
        # 3. Handle conflicting tweets
        conflict_tweets = df[df.duplicated(subset='tweet', keep=False)]
        if self.conflict_policy == "drop":
            df = df[~df['tweet'].isin(conflict_tweets['tweet'])]
        elif self.conflict_policy == "keep":
            df = conflict_tweets
            
        # 4. Lowercase
        df['tweet'] = df['tweet'].apply(lambda x: x.lower())
        
        # 5. Remove <user> and <url>
        df['tweet'] = df['tweet'].str.replace('<user>', '', regex=False)
        df['tweet'] = df['tweet'].str.replace('<url>', '', regex=False)
        
        # 6. Whitespace Stripping
        df['tweet'] = df['tweet'].apply(lambda x: x.strip())
        df['tweet'] = df['tweet'].apply(lambda x: " ".join(x.split()))
        
        # 7. Expand contractions
        df['tweet'] = df['tweet'].apply(contractions.fix)

        # 8.1 De-emojize [Creativity]
        df['tweet'] = df['tweet'].apply(lambda x: emoji.demojize(x, delimiters=(" ", " ")))
        df['tweet'] = df['tweet'].replace(":", "").replace("_", " ")
        
        # 8.2 De-emoticonize [Creativity]
        pattern = re.compile('|'.join(map(re.escape, emoticon_meanings.keys())))
        df['tweet'] = df['tweet'].apply(lambda tweet: pattern.sub(lambda x: emoticon_meanings[x.group()], tweet))
        
        '''
        [//TODO IN ORDER]
        > Handling Slang. Note: augment vocabulary by inspecting data
        
        > Stop-word removal
        
        > Handling Numerical values (remove)
        
        > Handle hashtag [creativity]: remove symbol and, for each hashtag, split it into the comprising words. 
        Then new columns? Append to tweet? For sure they are useful for sercasm detection
        Attenzione perchè qua, per ogni twwt, ogni hashtag devi separarlo nelle sue parole componenti. Vedi wordsegment
        #df['hashtags'] = df['tweet'].apply(lambda x: re.findall(r"#(\S+)", x))
        #df['tweet'] = df['tweet'].apply(lambda x: re.sub(r"#\S+", "", x))

        > Spelling correction (multiple letters, switched letters)
        
        > Remove Punctuation
        
        > Non-words / short/rare words removal
        
        > Part-of-Speech tagging
        
        > Lemmatization
        
        > Sarcasm detection [creativity] => change sentiment polarity (heuristic or DL?)
        
        > Dimensionality Reduction [creativity]
        
        > Text encoding/ Vectorization
        
        > Label Encoding
        
    
        Backlog:
            - Si possono mettere parole chiave come alert? Tipo 'guerra'? O è cheating?
            - Padding/Truncation
        '''

        return df
        

