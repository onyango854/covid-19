# data_cleaning.py
import pandas as pd
import numpy as np
from datetime import datetime

def clean_data(df):
    """Clean and prepare the dataset for analysis"""
    print("="*50)
    print("DATA CLEANING AND PREPARATION")
    print("="*50)
    
    # Create a copy of the dataframe
    df_clean = df.copy()
    
    # Handle publication date
    print("Processing publication dates...")
    df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')
    df_clean['publication_year'] = df_clean['publish_time'].dt.year
    
    # Fill missing years with a specific value (e.g., 0) for filtering
    df_clean['publication_year'] = df_clean['publication_year'].fillna(0).astype(int)
    
    # Create abstract word count
    print("Calculating abstract word counts...")
    df_clean['abstract_word_count'] = df_clean['abstract'].apply(
        lambda x: len(str(x).split()) if pd.notnull(x) else 0
    )
    
    # Create title word count
    df_clean['title_word_count'] = df_clean['title'].apply(
        lambda x: len(str(x).split()) if pd.notnull(x) else 0
    )
    
    # Handle missing values in key columns
    print("Handling missing values...")
    
    # For analysis purposes, we'll keep rows with at least title or abstract
    initial_count = len(df_clean)
    df_clean = df_clean[df_clean['title'].notna() | df_clean['abstract'].notna()]
    final_count = len(df_clean)
    
    print(f"Removed {initial_count - final_count} rows with both title and abstract missing")
    print(f"Final dataset size: {len(df_clean)} rows")
    
    # Fill other missing values with placeholder text
    df_clean['journal'] = df_clean['journal'].fillna('Unknown Journal')
    df_clean['authors'] = df_clean['authors'].fillna('Unknown Authors')
    
    return df_clean

def prepare_analysis_data(df_clean):
    """Prepare specific datasets for analysis"""
    
    # Papers by year (excluding years with value 0)
    yearly_counts = df_clean[df_clean['publication_year'] > 0]['publication_year'].value_counts().sort_index()
    
    # Top journals
    top_journals = df_clean['journal'].value_counts().head(20)
    
    # Word frequency in titles
    all_titles = ' '.join(df_clean['title'].dropna().astype(str))
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_titles.lower())
    word_freq = Counter(words).most_common(50)
    
    return yearly_counts, top_journals, word_freq

if __name__ == "__main__":
    # This would be called after data exploration
    import data_exploration
    
    file_path = "metadata.csv"
    df = data_exploration.load_data(file_path)
    
    if df is not None:
        df_clean = clean_data(df)
        yearly_counts, top_journals, word_freq = prepare_analysis_data(df_clean)
        
        print(f"Yearly counts: {yearly_counts}")
        print(f"Top 5 journals: {top_journals.head()}")
        print(f"Top 10 words in titles: {word_freq[:10]}")