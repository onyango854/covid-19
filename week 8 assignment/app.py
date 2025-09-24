# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon=":microscope:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file_path):
    """Load and cache the dataset"""
    try:
        df = pd.read_csv(file_path)
        st.sidebar.success("Data loaded successfully!")
        
        # Show available columns in sidebar for debugging
        st.sidebar.write("**Available columns:**")
        st.sidebar.write(list(df.columns))
        
        return df
    except FileNotFoundError:
        st.error(f"File {file_path} not found. Please make sure the file is in the correct location.")
        return None

def detect_date_column(df):
    """Detect which column contains date information"""
    date_columns = ['publish_time', 'publish_date', 'date', 'publication_date', 'time', 'year']
    
    for col in date_columns:
        if col in df.columns:
            return col
    
    # If no standard date column found, look for columns that might contain dates
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'publish']):
            return col
    
    return None

def detect_column_aliases(df):
    """Detect alternative column names for common fields"""
    column_mapping = {}
    
    # Date column
    date_col = detect_date_column(df)
    if date_col:
        column_mapping['publish_time'] = date_col
    else:
        column_mapping['publish_time'] = None
    
    # Journal column
    journal_cols = ['journal', 'journal_title', 'source', 'journal_name']
    column_mapping['journal'] = next((col for col in journal_cols if col in df.columns), None)
    
    # Title column
    title_cols = ['title', 'paper_title', 'document_title']
    column_mapping['title'] = next((col for col in title_cols if col in df.columns), None)
    
    # Abstract column
    abstract_cols = ['abstract', 'summary', 'paper_abstract']
    column_mapping['abstract'] = next((col for col in abstract_cols if col in df.columns), None)
    
    # Authors column
    author_cols = ['authors', 'author', 'authors_list']
    column_mapping['authors'] = next((col for col in author_cols if col in df.columns), None)
    
    # Source column
    source_cols = ['source_x', 'source', 'database', 'origin']
    column_mapping['source_x'] = next((col for col in source_cols if col in df.columns), None)
    
    return column_mapping

@st.cache_data
def clean_data(df, column_mapping):
    """Clean and prepare the data using detected column names"""
    df_clean = df.copy()
    
    # Handle publication date
    date_col = column_mapping['publish_time']
    if date_col:
        df_clean['publish_time'] = pd.to_datetime(df_clean[date_col], errors='coerce')
        df_clean['publication_year'] = df_clean['publish_time'].dt.year.fillna(0).astype(int)
        st.sidebar.write(f"Using date column: {date_col}")
    else:
        # If no date column found, create a dummy year column
        df_clean['publication_year'] = 2020  # Default year for COVID papers
        st.sidebar.warning("No date column found. Using default year 2020.")
    
    # Create word counts for abstract and title
    abstract_col = column_mapping['abstract']
    if abstract_col and abstract_col in df_clean.columns:
        df_clean['abstract_word_count'] = df_clean[abstract_col].apply(
            lambda x: len(str(x).split()) if pd.notnull(x) else 0
        )
    else:
        df_clean['abstract_word_count'] = 0
    
    title_col = column_mapping['title']
    if title_col and title_col in df_clean.columns:
        df_clean['title_word_count'] = df_clean[title_col].apply(
            lambda x: len(str(x).split()) if pd.notnull(x) else 0
        )
    else:
        df_clean['title_word_count'] = 0
    
    # Handle missing values in key columns
    initial_count = len(df_clean)
    
    # Keep rows that have at least title or abstract
    has_title = title_col and title_col in df_clean.columns
    has_abstract = abstract_col and abstract_col in df_clean.columns
    
    if has_title and has_abstract:
        df_clean = df_clean[df_clean[title_col].notna() | df_clean[abstract_col].notna()]
    elif has_title:
        df_clean = df_clean[df_clean[title_col].notna()]
    elif has_abstract:
        df_clean = df_clean[df_clean[abstract_col].notna()]
    
    final_count = len(df_clean)
    st.sidebar.write(f"Removed {initial_count - final_count} rows with missing data")
    st.sidebar.write(f"Final dataset size: {len(df_clean)} rows")
    
    # Fill other missing values with placeholder text
    journal_col = column_mapping['journal']
    if journal_col and journal_col in df_clean.columns:
        df_clean[journal_col] = df_clean[journal_col].fillna('Unknown Journal')
    
    authors_col = column_mapping['authors']
    if authors_col and authors_col in df_clean.columns:
        df_clean[authors_col] = df_clean[authors_col].fillna('Unknown Authors')
    
    # Store column mapping in the dataframe for later use
    df_clean.attrs['column_mapping'] = column_mapping
    
    return df_clean

def get_word_frequency(df, title_col, top_n=20):
    """Calculate word frequency from titles"""
    if title_col and title_col in df.columns:
        all_titles = ' '.join(df[title_col].dropna().astype(str))
        # Extract words (3+ letters only to avoid short/common words)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_titles.lower())
        word_freq = Counter(words).most_common(top_n)
        return word_freq
    return []

def main():
    # Header
    st.markdown('<div class="main-header">CORD-19 COVID-19 Research Data Explorer</div>', 
                unsafe_allow_html=True)
    
    st.write("""
    This application explores the CORD-19 dataset, which contains metadata about COVID-19 research papers.
    Use the filters in the sidebar to customize the analysis.
    """)
    
    # Load data
    file_path = "metadata.csv"
    df = load_data(file_path)
    
    if df is None:
        st.stop()
    
    # Detect column names
    column_mapping = detect_column_aliases(df)
    
    # Display detected columns
    st.sidebar.write("**Detected columns:**")
    for key, value in column_mapping.items():
        st.sidebar.write(f"{key}: {value}")
    
    # Clean data
    df_clean = clean_data(df, column_mapping)
    
    # Sidebar
    st.sidebar.title("Filters and Controls")
    
    # Year range filter
    valid_years = df_clean[df_clean['publication_year'] > 0]['publication_year']
    if len(valid_years) > 0:
        min_year = int(valid_years.min())
        max_year = int(valid_years.max())
    else:
        min_year = 2019
        max_year = 2022
    
    year_range = st.sidebar.slider(
        "Select publication year range:",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
    
    # Journal filter
    journal_col = column_mapping['journal']
    if journal_col and journal_col in df_clean.columns:
        top_journals = df_clean[journal_col].value_counts().head(20).index.tolist()
        selected_journals = st.sidebar.multiselect(
            "Filter by journal (top 20 only):",
            options=top_journals,
            default=[]
        )
    else:
        selected_journals = []
        st.sidebar.warning("No journal column found.")
    
    # Abstract word count filter
    min_words, max_words = st.sidebar.slider(
        "Abstract word count range:",
        min_value=0,
        max_value=int(df_clean['abstract_word_count'].max()) if len(df_clean) > 0 else 500,
        value=(0, 500)
    )
    
    # Apply filters
    filtered_df = df_clean[
        (df_clean['publication_year'] >= year_range[0]) & 
        (df_clean['publication_year'] <= year_range[1]) &
        (df_clean['abstract_word_count'] >= min_words) &
        (df_clean['abstract_word_count'] <= max_words)
    ]
    
    if selected_journals and journal_col:
        filtered_df = filtered_df[filtered_df[journal_col].isin(selected_journals)]
    
    # Display dataset info
    st.sidebar.write(f"**Filtered dataset:** {len(filtered_df)} papers")
    st.sidebar.write(f"**Original dataset:** {len(df_clean)} papers")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
        
        # Key metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Total Papers", len(filtered_df))
        
        with metric_col2:
            st.metric("Years Covered", f"{year_range[0]} - {year_range[1]}")
        
        with metric_col3:
            avg_words = filtered_df['abstract_word_count'].mean() if len(filtered_df) > 0 else 0
            st.metric("Avg Abstract Words", f"{avg_words:.0f}")
        
        with metric_col4:
            if journal_col and journal_col in filtered_df.columns:
                journals_count = filtered_df[journal_col].nunique() if len(filtered_df) > 0 else 0
            else:
                journals_count = 0
            st.metric("Unique Journals", journals_count)
        
        # Data sample
        st.write("Sample of the data (first 10 rows):")
        if len(filtered_df) > 0:
            # Show available columns in the sample
            sample_columns = []
            for col in ['title', 'journal', 'publication_year', 'abstract_word_count']:
                mapped_col = column_mapping.get(col, col)
                if mapped_col in filtered_df.columns:
                    sample_columns.append(mapped_col)
            
            if sample_columns:
                st.dataframe(filtered_df[sample_columns].head(10))
            else:
                st.dataframe(filtered_df.head(10))
        else:
            st.write("No data available for the selected filters.")
    
    with col2:
        st.markdown('<div class="section-header">Quick Facts</div>', unsafe_allow_html=True)
        
        if len(filtered_df) > 0:
            # Top journals in filtered data
            if journal_col and journal_col in filtered_df.columns:
                top_journals_filtered = filtered_df[journal_col].value_counts().head(5)
                st.write("**Top 5 Journals:**")
                for journal, count in top_journals_filtered.items():
                    st.write(f"- {journal}: {count} papers")
            
            # Year with most publications
            if 'publication_year' in filtered_df.columns:
                year_counts = filtered_df['publication_year'].value_counts()
                if len(year_counts) > 0:
                    year_most_pubs = year_counts.index[0]
                    count_most_pubs = year_counts.iloc[0]
                    st.write(f"**Year with most publications:** {year_most_pubs} ({count_most_pubs} papers)")
        else:
            st.write("No data available for the selected filters.")
    
    # Visualizations
    st.markdown('<div class="section-header">Visualizations</div>', unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Publications Over Time", 
        "Journal Analysis", 
        "Word Frequency", 
        "Source Analysis"
    ])
    
    with tab1:
        st.subheader("Publications by Year")
        if len(filtered_df) > 0 and 'publication_year' in filtered_df.columns:
            yearly_counts = filtered_df[filtered_df['publication_year'] > 0]['publication_year'].value_counts().sort_index()
            
            if len(yearly_counts) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(yearly_counts.index, yearly_counts.values, color='skyblue', alpha=0.7)
                ax.set_xlabel('Year')
                ax.set_ylabel('Number of Publications')
                ax.set_title('COVID-19 Research Publications Over Time')
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig)
            else:
                st.write("No valid year data available.")
        else:
            st.write("No data available for the selected filters.")
    
    with tab2:
        st.subheader("Top Publishing Journals")
        if len(filtered_df) > 0 and journal_col and journal_col in filtered_df.columns:
            top_journals_viz = filtered_df[journal_col].value_counts().head(15)
            
            if len(top_journals_viz) > 0:
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.barh(range(len(top_journals_viz)), top_journals_viz.values)
                ax.set_yticks(range(len(top_journals_viz)))
                # Truncate long journal names for display
                labels = [label[:50] + '...' if len(label) > 50 else label for label in top_journals_viz.index]
                ax.set_yticklabels(labels, fontsize=10)
                ax.set_xlabel('Number of Publications')
                ax.set_title('Top 15 Journals by Publication Count')
                ax.grid(axis='x', alpha=0.3)
                st.pyplot(fig)
            else:
                st.write("No journal data available.")
        else:
            st.write("No journal data available for the selected filters.")
    
    with tab3:
        st.subheader("Most Frequent Words in Paper Titles")
        
        title_col = column_mapping['title']
        if len(filtered_df) > 0 and title_col and title_col in filtered_df.columns:
            word_freq = get_word_frequency(filtered_df, title_col, top_n=15)
            
            if word_freq:
                words = [word for word, count in word_freq]
                counts = [count for word, count in word_freq]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(range(len(words)), counts, color='lightgreen', alpha=0.7)
                ax.set_yticks(range(len(words)))
                ax.set_yticklabels(words, fontsize=12)
                ax.set_xlabel('Frequency')
                ax.set_title('Top 15 Words in Paper Titles')
                ax.grid(axis='x', alpha=0.3)
                st.pyplot(fig)
                
                # Also display as a table
                st.subheader("Word Frequency Table")
                word_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
                st.dataframe(word_df)
            else:
                st.write("No title data available for word frequency analysis.")
        else:
            st.write("No title data available for the selected filters.")
    
    with tab4:
        st.subheader("Publications by Source")
        source_col = column_mapping['source_x']
        if len(filtered_df) > 0 and source_col and source_col in filtered_df.columns:
            source_counts = filtered_df[source_col].value_counts().head(10)
            
            if len(source_counts) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%', startangle=90)
                ax.set_title('Top 10 Sources of Publications')
                st.pyplot(fig)
            else:
                st.write("No source data available.")
        else:
            st.write("No source data available for the selected filters.")
    
    # Additional analysis
    st.markdown('<div class="section-header">Additional Analysis</div>', unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Abstract Length Distribution")
        if len(filtered_df) > 0 and filtered_df['abstract_word_count'].sum() > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(filtered_df[filtered_df['abstract_word_count'] > 0]['abstract_word_count'], 
                    bins=30, color='lightcoral', alpha=0.7)
            ax.set_xlabel('Word Count')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Abstract Lengths')
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        else:
            st.write("No abstract data available for the selected filters.")
    
    with col4:
        st.subheader("Papers with Abstracts")
        abstract_col = column_mapping['abstract']
        if len(filtered_df) > 0 and abstract_col and abstract_col in filtered_df.columns:
            has_abstract = filtered_df[abstract_col].notna().value_counts()
            
            if len(has_abstract) > 0:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.pie(has_abstract.values, labels=['With Abstract', 'Without Abstract'], 
                       autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
                ax.set_title('Papers with Available Abstracts')
                st.pyplot(fig)
            else:
                st.write("No abstract data available.")
        else:
            st.write("No abstract data available for the selected filters.")
    
    # Debug information (collapsible)
    with st.expander("Debug Information"):
        st.write("**Original DataFrame columns:**")
        st.write(list(df.columns))
        st.write("**Column mapping:**")
        st.write(column_mapping)
        st.write("**Cleaned DataFrame info:**")
        st.write(f"Shape: {df_clean.shape}")
        st.write("Columns in cleaned data:")
        st.write(list(df_clean.columns))
    
    # Footer
    st.markdown("---")
    st.write("Data Source: CORD-19 Research Dataset from Kaggle")
    st.write("This is a simplified analysis for educational purposes.")

if __name__ == "__main__":
    main()