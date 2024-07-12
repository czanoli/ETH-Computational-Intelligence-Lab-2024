import warnings
import gdown
import logging
import zipfile
import os
warnings.filterwarnings("ignore", category=UserWarning)

# URL of files hosted on Google Drive
fasttext_url = "https://drive.google.com/file/d/1wLUWaJNba-1403kXAprLmbVnXuDNEGW1/view?usp=sharing"
glove_url = "https://drive.google.com/file/d/1I-c7ERB9H1i5ifPwXFDTnas2LjcfLOy9/view?usp=sharing"
fasttext_file_name = "fastText-0.9.2.zip"
glove_file_name = "glove.twitter.27B.zip"
output_base_path = "src/models/"

# Setup logging
log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)

def download_file(url, name, output_path):
    """
    Download a file from a given URL and save it to the specified output path.

    Parameters
    ----------
    url : str
        The URL of the file to download.
    name : str
        The name of the file being downloaded (for logging purposes).
    output_path : str
        The path where the downloaded file will be saved.
    """
    try:
        logger.info(f"Downloading {name}...")
        gdown.download(url, output_path, quiet=False, fuzzy=True)
        logger.info("Download completed successfully.")
    except Exception as e:
        logger.error(f"Error during download of {name}: {e}")

def unzip_file(name, zip_path, extract_to):
    """
    Unzip a file to the specified directory.

    Parameters
    ----------
    name : str
        The name of the file being unzipped (for logging purposes).
    zip_path : str
        The path to the zip file.
    extract_to : str
        The directory where the contents of the zip file will be extracted.
    """
    try:
        logger.info(f"Unzipping file {zip_path} to {extract_to}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info("Unzipping completed successfully.")
    except Exception as e:
        logger.error(f"Error during unzipping of {name}: {e}")

if __name__ == "__main__":
    # FastText
    download_file(fasttext_url, 'fastText', output_base_path + fasttext_file_name)
    unzip_file('FastText', output_base_path + fasttext_file_name, output_base_path)
    
    # GloVe
    download_file(glove_url, 'GloVe', output_base_path + glove_file_name)
    unzip_file('GloVe', output_base_path + glove_file_name, output_base_path)
    
