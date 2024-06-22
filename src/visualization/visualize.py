import pandas as pd
import yaml
from pathlib import Path

# Load configuration from config.yaml
with open(Path(__file__).resolve().parent/'config.yml', 'r') as file:
    config = yaml.safe_load(file)

class Visualizer:
    def __init__(self):
        pass
