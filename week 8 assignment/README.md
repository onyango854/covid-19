# CORD-19 Data Analysis and Streamlit Application

This project analyzes the CORD-19 dataset containing metadata about COVID-19 research papers and presents the findings through an interactive Streamlit application.
## Project Structure
Frameworks_Assignment/
│
├── app.py # Main Streamlit application
├── data_exploration.py # Data loading and exploration
├── data_cleaning.py # Data cleaning and preparation
├── analysis_visualization.py # Data analysis and visualization
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── metadata.csv # Dataset (not included in repo due to size)

## Installation and Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd Frameworks_Assignment

Install required packages:
pip install -r requirements.txt
Download the metadata.csv file from the CORD-19 dataset on Kaggle and place it in the project directory.

Running the Streamlit Application
streamlit run app.py
Running Individual Analysis Scripts
python data_exploration.py
python data_cleaning.py
python analysis_visualization.py
