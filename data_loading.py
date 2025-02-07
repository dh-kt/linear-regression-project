#data_loading file
import pandas as pd

def load_data(filepath):
    """Load the dataset from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        print("Dataset loaded successfully.")
        print("First few rows of the dataset:")
        print(df.head())
        print("Data types:")
        print(df.dtypes)
        return df
    except FileNotFoundError:
        print(f'Error: The file not found at {filepath}.')
        return None
    except Exception as e:
        print(f'Error loading dataset: {e}.')
        return None