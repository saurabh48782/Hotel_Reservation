import pandas as pd


def read_csv_files(filepath: str) -> pd.DataFrame:
    '''Read csv file from a path and return a dataframe
    '''
    return pd.read_csv(filepath)
