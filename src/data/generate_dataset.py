import pandas as pd
import yaml
from pathlib import Path
from ydata_profiling import ProfileReport
import re
import contractions
import emoji
from spellchecker import SpellChecker
from collections import defaultdict


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

        #NOTE: I THINK PUNCTUATION SHOULD BE REMOVED BEFORE ALL THIS STUFF
        #NOTE: I 'lol' non vengono rimossi tutti perche a volte sono scritti male quindi -> prima di togliere parole vanno tolti errori
        #NOTE: stessa cosa per gli hashtag -> prima
        '''
            new order:
            9. togliere hashtags -> rimuovi simbolo, se underscore spezza, se camelcase spezza, altrimenti cerca di spezzare in parole
            10. toglere punteggiatura, simboli, numeri e spazi in eccesso
            11. rimuovere errori spelling
            12. fix slang e stop-words '''
        
        # 9. Hashtag removal
        def process_hashtags(tweet):
            words = tweet.split()
            new_words = []
            for word in words:
                if word.startswith('#'):
                    # if underscores
                    words = word[1:].replace('_', ' ')
                    # if camel-case
                    words = re.sub(r'([a-z])([A-Z])', r'\1 \2', words)
                    # if all undercase words -> common words list
                    split_words = []
                    temp_word = ''
                    i = 0
                    while i < len(words):
                        temp_word += words[i]
                        if temp_word.lower() in common_words:
                            split_words.append(temp_word)
                            temp_word = ''
                        i += 1
                    if temp_word:
                        split_words.append(temp_word)
                    word = ' '.join(split_words).lower()
                new_words.append(word)
            new_tweet = ' '.join(new_words)
            return new_tweet
        df['tweet'] = df['tweet'].apply(lambda x: process_hashtags(x))
        print('-------------------------------- hashtag removal complete')

        # 10. remove punctuation, symbols, digits and useless spaces + remove rt = retweet 
        def remove_punctuation_symbols_digits_spaces(tweet):
            tweet = re.sub(r'[^\w\s]', ' ', tweet)
            tweet = re.sub(r'\d', '', tweet)
            tweet = re.sub(r'\brt\b', '', tweet, flags=re.IGNORECASE).strip()
            tweet = ' '.join(tweet.split())
            return tweet
        df['tweet'] = df['tweet'].apply(lambda x: remove_punctuation_symbols_digits_spaces(x))
        print('-------------------------------- remove punctuation, symbols, digits and spaces complete')

        # 11. correct spelling
        spell = SpellChecker()
        def correct_spelling(tweet):
            words = tweet.split()
            reduced_words = [re.sub(r'(.)\1+', r'\1\1', word) for word in words]
            
            # try 1
            #corrected_words = [spell.correction(word) or word for word in reduced_words]
            # ci mette un botto cazzo

            # try 2 (uguale ma con cache)
            #corrected_words = []
            #correction_cache = {}
            #for word in reduced_words:
            #    if word in correction_cache:
            #        corrected_words.append(correction_cache[word])
            #    else:
            #        corrected_word = spell.correction(word) or word
            #        correction_cache[word] = corrected_word
            #        corrected_words.append(corrected_word)
            #return ' '.join(corrected_words)
            return ' '.join(reduced_words)
        df['tweet'] = df['tweet'].apply(lambda x: correct_spelling(x))
        print('-------------------------------- spelling check complete')

        # 12. replace slang and stopwords
        def replace_slang_and_remove_stopwords(tweet, slang_dict, stopwords):
            words = tweet.split()
            new_words = []
            for word in words:
                word_clean = re.sub(r'[^\w\s]', '', word)
                if word_clean.lower() in slang_dict:
                    new_word = slang_dict[word_clean.lower()]
                else:
                    new_word = word_clean
                if new_word.lower() not in stopwords:
                    new_words.append(new_word)
            new_tweet = ' '.join(new_words)
            return new_tweet
        df['tweet'] = df['tweet'].apply(lambda x: replace_slang_and_remove_stopwords(x, slang_dict, stopwords))
        print('-------------------------------- splanc and stopwords removal complete')

        



        
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
        

