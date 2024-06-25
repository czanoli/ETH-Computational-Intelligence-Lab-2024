import pandas as pd
import yaml
from pathlib import Path
from ydata_profiling import ProfileReport
import re
import contractions
import emoji
import nltk
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
from vocabulary import *
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
import wordninja
from symspellpy import SymSpell
import pkg_resources
from tqdm import tqdm

# Load configuration from config.yaml
with open(Path(__file__).resolve().parent/'config.yml', 'r') as file:
    config = yaml.safe_load(file)
train_files = config['raw_train_paths']
test_file = config['raw_test_path']
nltk.download('stopwords')
nltk.download('words')
common = words.words()
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
tqdm.pandas()


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
            cosa c'è:
            9. togliere hashtags -> rimuovi simbolo e spezza
            10. toglere punteggiatura, simboli, numeri e spazi in eccesso, lettere ripetute
            12. togliere slang (prima di spelling perche riduce il numero di parole mispelled, molto ricco di slang twitter)
            13. rimuovere errori spelling
            12. togliere stop-words'''
        
        # 9. Hashtag removal
        def process_hashtags(tweet):
            '''This function addresses the hashtag removal task.
                Identifies hasthags, removes hashtag's symbol and split content into words'''
            # 9.1 - split tweet
            words = tweet.split()
            # 9.2 - iterate over words
            new_words = []
            for word in words:
                # 9.3 - if word begins with hashtag
                if word.startswith('#'):
                    # 9.4 - split hashtag into list of single words 
                    words = wordninja.split(word.lstrip("#"))
                    # 9.5 - join list of words into single string
                    split_words = " ".join(words).lower()
                    # 9.6 - append to list of words of the tweet
                    new_words.append(split_words)
                else:
                    # 9.7 - if not hashtag simply add words to the list of words of the tweet
                    new_words.append(word)
            # 9.8 - merge all words of the tweet together
            new_tweet = ' '.join(new_words)
            return new_tweet
        df['tweet'] = df['tweet'].apply(lambda x: process_hashtags(x))
        print('-------------------------------- hashtag removal completed')

        # 10. tweet cleaning
        def remove_punctuation_symbols_digits_spaces(tweet):
            ''''This function does a general cleaning of the tweets.
                Operations:
                    - put tweet into lowercase
                    - remove retweet symbol
                    - remove digits and symbols
                    - remove extra blank spaces
                    - remove single letters
                    - reduce repeated letters
                    - replace 'ahahaha' with 'laugh'
                    - replace 'xoxo' with 'kiss'
                    '''
            # 10.1 - put in lower case
            tweet = tweet.lower()
            # 10.2 - remove retweet symbol 'rt'
            tweet = re.sub(r'\brt\b', '', tweet)
            # 10.3 - keep only letters and blank spaces
            tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)
            # 10.4 - remove extra blank spaces
            tweet = re.sub(r'\s+', ' ', tweet).strip()
            # 10.5 - remove single letter (mostly errors or 'I' which are not useful)
            tweet = re.sub(r'\b\w\b', '', tweet)
            # 10.6 - reduce repeated letters at the end of the word
            tweet = re.sub(r'(\w*?)(\w)(?![ls])\2{1,}', r'\1\2', tweet)
            # 10.7 - replace 'ahahahah' and similar with 'laugh'
            tweet = re.sub(r'\b(?:[ah]*h[ah]*){2,}\b', 'laugh', tweet)
            # 10.8 - replace 'xoxo' and similar with 'kiss'
            tweet = re.sub(r'\b(xo)+x?\b', 'kiss', tweet)
            return tweet
        df['tweet'] = df['tweet'].apply(lambda x: remove_punctuation_symbols_digits_spaces(x))
        print('-------------------------------- cleaning completed')

        # 11. replace slang
        def replace_slang(tweet, slang_dict):
            '''This function replaces slang words with regular expressions
                For each word in the tweet it checks if it belongs to the slang dictionary and if it does the word gets replaced '''
            # 11.1 - split tweet
            words = tweet.split()
            # 11.2 - iterate over tweet's words
            new_words = []
            for word in words:
                # 11.3 - if words belongs to the slang dictionary
                if word.lower() in slang_dict:
                   # 11.4 - word gets replaced by its regular corresponding
                    new_word = slang_dict[word.lower()]
                else:
                    # 11.5 - else keep word unchanged
                    new_word = word
                # 11.6 - append word to list of words of the tweet
                new_words.append(new_word)
            # 11.7 - merge all words of the tweet together
            new_tweet = ' '.join(new_words)
            return new_tweet
        df['tweet'] = df['tweet'].apply(lambda x: replace_slang(x, slang_dict))
        print('-------------------------------- slang replacement completed')

        # 12. correct spelling
        def correct_spelling(tweet):
            '''This function corrects the spelling of the words.
                Using SymSpell algorithm to correct spelling, if no correction is found the original spelling is kept'''
            # 12.1 - get suggestions for the input tweet
            suggestions = sym_spell.lookup_compound(tweet, max_edit_distance=2, transfer_casing=True)
            # 12.2 - if there is a suggestion
            if suggestions:
                # 12.3 - take the closest suggestion
                tweet = suggestions[0].term
            # 12.4 - if there is no suggestion keep the spelling unchanged
            return tweet
        df['tweet'] = df['tweet'].progress_apply(lambda x: correct_spelling(x))
        print('-------------------------------- spelling check completd')

        # 13. remove stopwords
        def remove_stopwords(tweet):
            '''This function removes stopwords from the tweet
                For each word in the tweet it checks if it belongs to the NLTK stopwords set and if it does the word gets replaced '''
            # 13.1 - set of english stopwords
            stop_words = set(stopwords.words('english'))
            # 13.2 - split tweet
            words = tweet.split()
            # 13.3 - iterate over each word of the tweet
            new_words = []
            for word in words:
                # 13.4 - if word is not found in the stopword set
                if word.lower() not in stop_words:
                    # 13.5 - append this word to the tweet words list
                    new_words.append(word)
            # 13.6 - merge all words of the tweet together
            new_tweet = ' '.join(new_words)
            return new_tweet
        df['tweet'] = df['tweet'].apply(lambda x: remove_stopwords(x))
        print('--------------------------------  stopwords removal completed')

        # 14. lemmatization
        # 14.1 - initialize the WordNet lemmatizer
        lemmatizer = WordNetLemmatizer()
        # 14.2 - assigns part-of-speech tags to each word
        def get_pos(treebank_tag):
            ''' Thos function converts Treebank POS tags (NLTK) to WordNet POS tags
                - Treebank POS Tags: tags used by the POS tagger in NLTK
                - WordNet POS Tags: tags used by the WordNet lemmatizer'''
            # 14.2.1 - case 1: adjective, map J -> Wordnet.ADJ 
            if treebank_tag.startswith('J'):
                return wordnet.ADJ
            # 14.2.2 - case 2: verb, map V -> Wordnet.VERB
            elif treebank_tag.startswith('V'):
                return wordnet.VERB
            # 14.2.3 - case 3: noun, map N -> Wordnet.NOUN 
            elif treebank_tag.startswith('N'):
                return wordnet.NOUN
            # 14.2.4 - case 4: adverb, map R -> Wordnet.ADV 
            elif treebank_tag.startswith('R'):
                return wordnet.ADV
            # 14.2.5 - case 5: if no match found default is NOUN
            else:
                return wordnet.NOUN
        # 14.3 - lemmatize each word in a tweet based on its POS tag
        def lemmatization(tweet):
            '''This function lemmatizes the words in a tweet based on their POS tags using WordNetLemmatizer'''
            # 14.3.1 - split tweet
            words = tweet.split()
            # 14.3.2 - get POS tags for each word
            pos_tags = pos_tag(words)
            # 14.3.3 - lemmatize each word with its POS tag
            lemmatized_words = [lemmatizer.lemmatize(word, get_pos(tag)) for word, tag in pos_tags]
            # 14.3.4 - merge all lemmatized words of the tweet together
            lemmatized_tweet = ' '.join(lemmatized_words)
            return lemmatized_tweet
        df['tweet'] = df['tweet'].apply(lambda x: lemmatization(x))
        print('-------------------------------- lemmatization complete')

        
        '''
        [//TODO IN ORDER]
            > [DONE] Handling Slang. Note: augment vocabulary by inspecting data
        
            > [DONE] Stop-word removal
        
            > [DONE] Handling Numerical values (remove)
        
            > [DONE] Handle hashtag [creativity]: remove symbol and, for each hashtag, split it into the comprising words. 
                        Then new columns? Append to tweet? For sure they are useful for sercasm detection
                        Attenzione perchè qua, per ogni twwt, ogni hashtag devi separarlo nelle sue parole componenti. Vedi wordsegment
                        #df['hashtags'] = df['tweet'].apply(lambda x: re.findall(r"#(\S+)", x))
                        #df['tweet'] = df['tweet'].apply(lambda x: re.sub(r"#\S+", "", x))

            > [DONE] Spelling correction (multiple letters, switched letters)
        
        
            > [DONE PARTIALLY -> only short words] Non-words / short/rare words removal
        
            > [DONE] Part-of-Speech tagging
        
            > [DONE] Lemmatization
        
        > Sarcasm detection [creativity] => change sentiment polarity (heuristic or DL?)
        
        > Dimensionality Reduction [creativity]
        
        > Text encoding/ Vectorization
        
        > Label Encoding
        
    
        Backlog:
            - Si possono mettere parole chiave come alert? Tipo 'guerra'? O è cheating?
            - Padding/Truncation
        '''

        return df
        

