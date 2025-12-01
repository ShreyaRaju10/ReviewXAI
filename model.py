#importing lib
import re
import nltk
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud


#downloading nltk data
def download_nltk_data():
    nltk.download('stopwords')
    nltk.download('punkt')
    all_stopwords = nltk.corpus.stopwords.words('english')
    return all_stopwords
stop_words = download_nltk_data()

# preprocessing 
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    words = text.split()
    text = ' '.join(words)
    return text

#sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0.1:
        sentiment = "Positive"
        color = '#8FC7A3'
    elif polarity < -0.1:
        sentiment = "Negative"
        color = '#8FC7A3'
    else:
        sentiment = "Neutral"
        color = '#8FC7A3'

    return sentiment, polarity, color

#extracting key words
def extract_topics(texts, n_topics=5):
    processed_texts = [preprocess_text(text) for text in texts]

    vectorizer = TfidfVectorizer(max_features = 100, stop_words = 'english', ngram_range=(1,2))

    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    feature_names = vectorizer.get_feature_names_out()

    tfidf_scores = tfidf_matrix.sum(axis=0).A1

    top_indices = tfidf_scores.argsort()[-n_topics: ][::-1]
    top_topics = [feature_names[i] for i in top_indices]
    return top_topics

#World Cloud
def generate_wordcloud(text):
    worldcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white', 
        max_words = 50,
        stopwords=stop_words, 
        colormap='viridis').generate(text)
    return worldcloud