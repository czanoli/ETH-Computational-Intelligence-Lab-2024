# -*- coding: utf-8 -*-
#import click
import os
import yaml
import logging
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv
from generate_dataset import DataProcessor

# Load configuration from config.yaml
with open(Path(__file__).resolve().parent/'config.yml', 'r') as file:
    config = yaml.safe_load(file)
dataset_type = config["dataset_type"]
train_files = config['raw_train_paths']
test_file = config['raw_test_path']
processed_path = config['processed_train_path']

#@click.command()
#@click.option('-i', '--input_datapath', type=click.Path(exists=True), required=True, help='Path to the folder containing raw training data.')
#@click.option('-t', '--data_type', type=click.Choice(['full', 'small'], case_sensitive=False), required=False, help='Dataset type to be loaded.')
#@click.option('-o', '--output_datapath', type=click.Path(), required=True, help='Path to the output folder where processed training data will be saved.')
def main():
    """ Runs data processing scripts to turn raw training data into
        cleaned training data ready to be analyzed.
    """
    logger = logging.getLogger(__name__)
    
    logger.info('Creating DataProcessor instance ...')
    data_processor = DataProcessor(train_files, test_file, processed_path)

    logger.info('Pre-processing Twitter training data from raw data ...')
    
    df = data_processor.load_training_data(dataset_type)
    
    # [TODO]: pre-processing
    
    data_processor.save_df_to_csv(df)
    
    logger.info(f'Training data has been pre-processed and saved at {processed_path} ! ')


if __name__ == '__main__':
    
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    project_dir = Path(__file__).resolve().parents[2]
    #load_dotenv(find_dotenv())

    main()

