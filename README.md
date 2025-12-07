# ReviewXAI - Sentiment & Topic Analysis Tool

ReviewXAI is an intelligent web application designed to analyse customer feedback and reviews. Leveraging Natural Language Processing (NLP) and Machine Learning techniques, it provides instant insights into sentiment polarity and extracts key topics from text data.
This project is labelled XAI (Explainable AI) because it moves beyond "Black Box" predictions. Instead of just giving you a final label (like "Positive" or "Negative"), it provides transparent evidence for why that result was chosen:
1. Quantitative Transparency: It displays the exact Polarity Score and Confidence %, showing the precise mathematical strength of the sentiment rather than a vague guess.
2. Contextual Explanation: By extracting Key Topics (TF-IDF) and generating Word Clouds, the tool highlights exactly which words and themes exist in the text, helping the user understand the context behind the data.

## Features

1. Single Review Analysis:
  Instant Sentiment classification (Positive, Neutral, Negative).
  Polarity confidence score with a dynamic Gauge Chart.
  Automatic keyword/topic extraction using TF-IDF.

2. Batch Analysis (CSV):
  Upload large datasets of reviews.
  Interactive visualizations: Sentiment Distribution (Pie Chart) and Polarity Histograms.
  Word Cloud generation to visualize the most frequent terms.

3. Custom UI: A sleek, dark-themed interface built with custom CSS for a modern user experience.

## Tech Stack
  1. Frontend: Streamlit
  2. NLP & ML: NLTK, TextBlob, Scikit-learn (TF-IDF)
  3. Data Manipulation: Pandas
  4. Visualization: Plotly, Matplotlib, WordCloud

## Project Structure
		ReviewXAI/
		│
		├── model.py             # Contains the NLP logic
		├── UI.py                # Streamlit frontend application
		├── requirements.txt     # List of dependencies
		├── Sample_Reviews.csv   # Sample dataset
		├── README.md            # Project documentation
		└── .streamlit/
		    └── config.toml      # Streamlit configuration file

## Installation & Setup
	1. Clone the repository
		git clone 
		cd ReviewXAI
	2. Install dependencies
		pip install -r requirements.txt
	3. Run App
		streamlit run app.py

## How It Works
* Preprocessing: Text is cleaned by removing special characters and converting to lowercase. Stopwords are removed using NLTK.
* Sentiment Analysis: Uses TextBlob to calculate a polarity score ranging from -1 (Negative) to +1 (Positive).
	> 0.1: Positive,  
	< -0.1: Negative,  
	Else: Neutral
* Topic Extraction: Uses TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer from Scikit-learn to identify significant n-grams (words and phrases) that define the context of the review.

## Usage Guide

* Single Review Mode
1. Select "Single Review" from the sidebar.
2. Paste a customer review into the text area.
3. Click Analyze to see the sentiment gauge and key topics.

* Batch Analysis Mode
1. Select "Batch Analysis (CSV)" from the sidebar.
2. Upload a .csv file containing your data.
3. Select the column that contains the text reviews.
4. Click Analyze All Reviews to generate the dashboard report.

## App Interface
![1](1.jpeg)
