# -*- coding: utf-8 -*-
import os
import yaml
import logging
from pathlib import Path
from generate_dataset import DataProcessor

# Load configuration from config.yaml
with open(Path(__file__).resolve().parent/'config.yml', 'r') as file:
    config = yaml.safe_load(file)

duplicates_policy = config["duplicates_policy"]
conflict_policy = config["conflict_policy"]
hashtag_policy = config["hashtag_policy"]
preprocessing_policy = config['preprocessing_policy_optimal']
train_dataset_type = config["train_dataset_type"]
train_files = config['raw_train_paths']
test_file = config['raw_test_path']
processed_train_path = config['processed_train_path']
processed_test_path = config['processed_test_path']

def main():
    """
    Runs data processing scripts to turn raw training and testing data 
    into cleaned versions ready to be analyzed.
    """
    logger = logging.getLogger(__name__)
    
    logger.info('Creating DataProcessor instance ...')
    logger.info(f"Using pre-processing policy {preprocessing_policy['name']} ...")
    data_processor = DataProcessor(duplicates_policy, conflict_policy, hashtag_policy, 
                                   train_dataset_type, project_dir, train_files, test_file, preprocessing_policy)

    logger.info('Loading training data ...')
    df = data_processor.load_data()
    logger.info('Training data loaded.')
    
    logger.info('Pre-processing training data from raw data ...')
    df = data_processor.process_dataframe(df)
    logger.info('Training data pre-processing completed.')
    
    print()
    print(df.head())
    print()
    
    logger.info('Saving pre-processed training data ...')
    data_processor.save_df_to_csv(df, processed_train_path)
    logger.info(f'Pre-processed training data saved at {processed_train_path}.')
         
    logger.info('Loading test data ...')
    df_test = data_processor.load_data(is_test=True)
    logger.info('Test data loaded.')
    
    logger.info('Pre-processing testing data from raw data ...')
    df_test = data_processor.process_dataframe(df_test)
    logger.info('Test data pre-processing completed.')
    
    logger.info('Saving pre-processed test data ...')
    data_processor.save_df_to_csv(df_test, processed_test_path)
    logger.info(f'Pre-processed test data saved at {processed_test_path}.')


if __name__ == '__main__':
    
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    project_dir = Path(__file__).resolve().parents[2]
    os.chdir(project_dir)
    main()

