import pandas as pd
from pathlib import Path

def load_108():
    path = Path(__file__).parent / 'curated-108.csv'
    with open(path, encoding='utf-8') as file:
        dataframe = pd.read_csv(file)
    return dataframe