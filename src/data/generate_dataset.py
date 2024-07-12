import pandas as pd
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

# Download necessary NLTK dependencies and load SymSpell configs
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('words')
common = words.words()
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
tqdm.pandas()


class DataProcessor:
    """
    Initialize the DataProcessor class with specified policies and file paths.

    Parameters
    ----------
    duplicates_policy : str
        Policy for handling duplicates (e.g., 'drop', 'keep').
    shared_duplicates_policy : str
        Policy for handling duplicates shared between datasets (e.g., 'drop', 'keep').
    conflict_policy : str
        Policy for handling conflicting tweets (e.g., 'drop', 'keep').
    hashtag_policy : str
        Policy for handling hashtags (e.g., 'keep', 'drop').
    dataset_type : str
        The type of dataset (e.g., 'full', 'small').
    prj_dir : str
        The project directory path.
    train_files : dict
        Dictionary containing paths to raw training data files.
    test_file : str
        The path to the raw test data file.
    preprocessing_policy : dict
        Dictionary specifying the preprocessing steps to be applied.
    """
    def __init__(self, duplicates_policy, conflict_policy, hashtag_policy,
                 dataset_type, prj_dir, train_files, test_file, preprocessing_policy):
        self.duplicates_policy = duplicates_policy
        self.conflict_policy = conflict_policy
        self.hashtag_policy = hashtag_policy
        self.dataset_type = dataset_type
        self.prj_dir = prj_dir
        self.train_files = train_files
        self.test_file = test_file
        self.preprocessing_policy = preprocessing_policy
    
    def load_data(self, is_test=False):
        """
        Load data from the specified files.

        Parameters
        ----------
        is_test : bool, optional
            Flag to indicate if loading test data. Default is False.

        Returns
        -------
        pd.DataFrame
            The loaded DataFrame.
        """ 
        if is_test:
            df = pd.DataFrame()
            with open(self.test_file, 'r', encoding='utf-8') as file:
                data = [line.split(',', 1) for line in file.readlines()]
            df = pd.DataFrame(data, columns=["id", "tweet"])
            return df
        else:
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
    
    def save_df_to_csv(self, df, output_file):
        """
        Save DataFrame to a CSV file.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be saved.
        output_file : str
            The path where the CSV file will be saved.
        """
        df.to_csv(output_file, index=False)
        
    def nulls_info(self, df):
        """
        Return the number of null values in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.

        Returns
        -------
        pd.Series
            The count of null values per column.
        """
        return df.isnull().sum()
    
    def process_dataframe(self, df):
        """
        Process the DataFrame based on the specified policy.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.

        Returns
        -------
        pd.DataFrame
            The processed DataFrame.
        """
        # 1. Handle null values
        if self.preprocessing_policy.get("handle_null"):
            df = df.dropna()
            print('-------------------------------- nulls removal completed')
        
        # 2. Handle duplicates
        if self.preprocessing_policy.get("handle_duplicates"):
            if self.duplicates_policy == "drop":
                df = df.drop_duplicates()
            elif self.duplicates_policy == "keep":
                df = df[df.duplicated(keep=False)]
            print('-------------------------------- duplicates handling completed')
            
        # 3. Handle conflicting tweets
        if self.preprocessing_policy.get("handle_conflicting_tweets"):
            conflict_tweets = df[df.duplicated(subset='tweet', keep=False)]
            if self.conflict_policy == "drop":
                df = df[~df['tweet'].isin(conflict_tweets['tweet'])]
            elif self.conflict_policy == "keep":
                df = conflict_tweets
            print('-------------------------------- conflicting tweets completed')
            
        # 4. Lowercase
        if self.preprocessing_policy.get("lowercasing"):
            df['tweet'] = df['tweet'].apply(lambda x: x.lower())
            print('-------------------------------- lowercasing completed')
        
        # 5. Remove <user> and <url>
        if self.preprocessing_policy.get("tag_removal"):
            df['tweet'] = df['tweet'].str.replace('<user>', '', regex=False)
            df['tweet'] = df['tweet'].str.replace('<url>', '', regex=False)
            print('-------------------------------- tag removal completed')
        
        # 6. Whitespace Stripping
        if self.preprocessing_policy.get("whitespace_stripping"):
            df['tweet'] = df['tweet'].apply(lambda x: x.strip())
            df['tweet'] = df['tweet'].apply(lambda x: " ".join(x.split()))
            print('-------------------------------- whitespace stripping completed')
        
        # 7. Expand contractions
        if self.preprocessing_policy.get("handle_contractions"):
            df['tweet'] = df['tweet'].apply(contractions.fix)
            print('-------------------------------- contractions handling completed')

        # 8.1 De-emojize
        if self.preprocessing_policy.get("de_emojze"):
            df['tweet'] = df['tweet'].apply(lambda x: emoji.demojize(x, delimiters=(" ", " ")))
            df['tweet'] = df['tweet'].replace(":", "").replace("_", " ")
            print('-------------------------------- de-emojization completed')
        
        # 8.2 De-emoticonize
        if self.preprocessing_policy.get("de_emoticonize"):
            pattern = re.compile('|'.join(map(re.escape, emoticon_meanings.keys())))
            df['tweet'] = df['tweet'].apply(lambda tweet: pattern.sub(lambda x: emoticon_meanings[x.group()], tweet))
            print('-------------------------------- de-emoticonization completed')
        
        # 9. Hashtag removal
        def process_hashtags(tweet):
            """
            Process hashtags by removing the hashtag symbol and splitting the content into meaningful words.

            Parameters
            ----------
            tweet : str
                The tweet to process.

            Returns
            -------
            str
                The processed tweet with hashtags handled.
            """
            words = tweet.split()
            new_words = []
            for word in words:
                if word.startswith('#'):
                    if self.hashtag_policy == "keep":
                        words = wordninja.split(word.lstrip("#"))
                        split_words = " ".join(words).lower()
                        new_words.append(split_words)
                    elif self.hashtag_policy == "drop":
                        words = tweet.split()
                        new_words = [word for word in words if not word.startswith('#')]
                        new_tweet = ' '.join(new_words)
                    else:
                        raise ValueError("Wrong hashtag_policy provided.")
                else:
                    new_words.append(word)
            new_tweet = ' '.join(new_words)
            return new_tweet
        
        if self.preprocessing_policy.get("hastag_handling"):
            df['tweet'] = df['tweet'].apply(lambda x: process_hashtags(x))
            print('-------------------------------- hashtag removal completed')

        # 10. tweet cleaning
        def remove_punctuation_symbols_digits_spaces(tweet):
            """
            General cleaning of tweets by removing some special symbols and extra spaces.

            Parameters
            ----------
            tweet : str
                The tweet to clean.

            Returns
            -------
            str
                The cleaned tweet.
            """
            # keep only letters, numbers, rt and specific symbols: (, ), rt, ?, !
            pattern = r'[^a-zA-Z0-9\(\)rt\?!\s]'
            tweet = re.sub(pattern, '', tweet)
            return tweet
        
        if self.preprocessing_policy.get("handle_punctuation"):
            df['tweet'] = df['tweet'].apply(lambda x: remove_punctuation_symbols_digits_spaces(x))
            print('-------------------------------- punctuation removal completed')

        # 11. replace slang
        def replace_slang(tweet, slang_dict):
            """
            Replace slang words in the tweet with their standard equivalents.

            Parameters
            ----------
            tweet : str
                The tweet to process.
            slang_dict : dict
                Dictionary of slang terms and their standard equivalents.

            Returns
            -------
            str
                The tweet with slang replaced.
            """
            words = tweet.split()
            new_words = []
            for word in words:
                if word.lower() in slang_dict:
                    new_word = slang_dict[word.lower()]
                else:
                    new_word = word
                new_words.append(new_word)
            new_tweet = ' '.join(new_words)
            return new_tweet
        
        if self.preprocessing_policy.get("replace_slang"):
            df['tweet'] = df['tweet'].apply(lambda x: replace_slang(x, slang_dict))
            print('-------------------------------- slang replacement completed')

        # 12. correct spelling
        def correct_spelling(tweet):
            """
            Correct spelling errors in the tweet using the SymSpell algorithm.

            Parameters
            ----------
            tweet : str
                The tweet to process.

            Returns
            -------
            str
                The tweet with spelling corrected.
            """
            suggestions = sym_spell.lookup_compound(tweet, max_edit_distance=2, transfer_casing=True)
            if suggestions:
                tweet = suggestions[0].term
            return tweet
        
        if self.preprocessing_policy.get("correct_spelling"):
            df['tweet'] = df['tweet'].progress_apply(lambda x: correct_spelling(x))
            print('-------------------------------- spelling check completd')

        # 13. remove stopwords
        def remove_stopwords(tweet):
            """
            Remove stopwords from the tweet.

            Parameters
            ----------
            tweet : str
                The tweet to process.

            Returns
            -------
            str
                The tweet with stopwords removed.
            """
            stop_words = set(stopwords.words('english'))
            words = tweet.split()
            new_words = []
            for word in words:
                if word.lower() not in stop_words:
                    new_words.append(word)
            new_tweet = ' '.join(new_words)
            return new_tweet
        
        if self.preprocessing_policy.get("remove_stopwords"):
            df['tweet'] = df['tweet'].apply(lambda x: remove_stopwords(x))
            print('--------------------------------  stopwords removal completed')

        # 14. lemmatization
        lemmatizer = WordNetLemmatizer()
        def get_pos(treebank_tag):
            """
            Convert Treebank POS tags to WordNet POS tags.

            Parameters
            ----------
            treebank_tag : str
                The Treebank POS tag.

            Returns
            -------
            str
                The corresponding WordNet POS tag.
            """
            if treebank_tag.startswith('J'):
                return wordnet.ADJ
            elif treebank_tag.startswith('V'):
                return wordnet.VERB
            elif treebank_tag.startswith('N'):
                return wordnet.NOUN
            elif treebank_tag.startswith('R'):
                return wordnet.ADV
            else:
                return wordnet.NOUN
 
        def lemmatization(tweet):
            """
            Lemmatize words in the tweet based on their POS tags.

            Parameters
            ----------
            tweet : str
                The tweet to lemmatize.

            Returns
            -------
            str
                The lemmatized tweet.
            """
            words = tweet.split()
            pos_tags = pos_tag(words)
            lemmatized_words = [lemmatizer.lemmatize(word, get_pos(tag)) for word, tag in pos_tags]
            lemmatized_tweet = ' '.join(lemmatized_words)
            return lemmatized_tweet
        
        if self.preprocessing_policy.get("lemmatization"):
            df['tweet'] = df['tweet'].apply(lambda x: lemmatization(x))
            print('-------------------------------- lemmatization completed')

        return df
        

