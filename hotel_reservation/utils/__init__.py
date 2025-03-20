import pandas as pd


def read_csv_files(filepath: str) -> pd.DataFrame:
    '''Read csv files from a filepath and returns a dataframe
    '''
    return pd.read_csv(filepath)
