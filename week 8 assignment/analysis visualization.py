# analysis_visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter

def create_visualizations(df_clean, yearly_counts, top_journals, word_freq):
    """Create all required visualizations"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CORD-19 Dataset Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Publications over time
    axes[0, 0].bar(yearly_counts.index, yearly_counts.values, color='skyblue')
    axes[0, 0].set_title('Number of Publications by Year')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Number of Publications')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Top publishing journals (top 10)
    top_10_journals = top_journals.head(10)
    axes[0, 1].barh(range(len(top_10_journals)), top_10_journals.values)
    axes[0, 1].set_yticks(range(len(top_10_journals)))
    axes[0, 1].set_yticklabels(top_10_journals.index, fontsize=8)
    axes[0, 1].set_title('Top 10 Publishing Journals')
    axes[0, 1].set_xlabel('Number of Publications')
    
    # Plot 3: Word cloud for titles
    all_titles = ' '.join(df_clean['title'].dropna().astype(str))
    wordcloud = WordCloud(width=400, height=300, background_color='white', 
                         max_words=100, colormap='viridis').generate(all_titles)
    axes[1, 0].imshow(wordcloud, interpolation='bilinear')
    axes[1, 0].set_title('Word Cloud of Paper Titles')
    axes[1, 0].axis('off')
    
    # Plot 4: Distribution of paper counts by source (simplified)
    source_counts = df_clean['source_x'].value_counts().head(10)
    axes[1, 1].bar(range(len(source_counts)), source_counts.values, color='lightgreen')
    axes[1, 1].set_xticks(range(len(source_counts)))
    axes[1, 1].set_xticklabels(source_counts.index, rotation=45, ha='right', fontsize=8)
    axes[1, 1].set_title('Top 10 Sources by Publication Count')
    axes[1, 1].set_ylabel('Number of Publications')
    
    plt.tight_layout()
    plt.savefig('cord19_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional visualization: Abstract word count distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df_clean[df_clean['abstract_word_count'] > 0]['abstract_word_count'], 
             bins=50, color='lightcoral', alpha=0.7)
    plt.title('Distribution of Abstract Word Counts')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.axvline(df_clean['abstract_word_count'].median(), color='red', linestyle='dashed', linewidth=1)
    plt.text(df_clean['abstract_word_count'].median()*1.1, plt.ylim()[1]*0.9, 
             f'Median: {df_clean["abstract_word_count"].median():.0f} words')
    plt.savefig('abstract_word_count.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

if __name__ == "__main__":
    # This would be called after data cleaning
    import data_exploration
    import data_cleaning
    
    file_path = "metadata.csv"
    df = data_exploration.load_data(file_path)
    
    if df is not None:
        df_clean = data_cleaning.clean_data(df)
        yearly_counts, top_journals, word_freq = data_cleaning.prepare_analysis_data(df_clean)
        
        create_visualizations(df_clean, yearly_counts, top_journals, word_freq)