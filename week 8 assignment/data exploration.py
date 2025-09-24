# data_exploration.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Load the data
def load_data(file_path):
    """Load the metadata.csv file"""
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully!")
        return df
    except FileNotFoundError:
        print(f"File {file_path} not found. Please check the file path.")
        return None

def basic_exploration(df):
    """Perform basic data exploration"""
    print("="*50)
    print("BASIC DATA EXPLORATION")
    print("="*50)
    
    # DataFrame dimensions
    print(f"Dataset shape: {df.shape}")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    
    # Column names and data types
    print("\nColumn information:")
    print(df.info())
    
    # First few rows
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Check for missing values
    print("\nMissing values by column:")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percent
    })
    print(missing_df.sort_values('Missing Count', ascending=False))
    
    # Basic statistics for numerical columns
    print("\nBasic statistics:")
    print(df.describe())
    
    return missing_df

if __name__ == "__main__":
    # Load and explore the data
    file_path = "metadata.csv"  # Update this path if needed
    df = load_data(file_path)
    
    if df is not None:
        missing_df = basic_exploration(df)