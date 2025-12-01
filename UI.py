import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Import func
from model import (
    preprocess_text,
    analyze_sentiment,
    extract_topics,
    generate_wordcloud,
)

#Page Config
st.set_page_config(
    page_title="Sentiment & Topic Classifier",
    page_icon="📊",
    layout="wide"
)

#CSS Code
custom_css = """
<style>

/* Main page bg  */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #001419, #003B46, #005F73) !important;
    background-attachment: fixed;
}

/* Side bar */
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #001419, #00323D) !important;
}

/* Text */
* {
    color: #E8FDF9 !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(90deg, #007F8C, #00C6A2) !important;
    color: white !important;
    border: none !important;
    padding: 0.6rem 1.2rem !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    transition: 0.3s !important;
}

/* hover */
.stButton > button:hover {
    background: linear-gradient(90deg, #00C6A2, #007F8C) !important;
    transform: scale(1.02);
}

</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

def main():
    st.title("📈 ReviewXAI")
    st.markdown('''<p style='font-size: 22px;'> Analyse customer reviews and feedback with AI-powered sentiment and topic analyser</p>''',
    unsafe_allow_html=True)
    
    st.sidebar.header("⚙️ Options")
    analysis_mode = st.sidebar.radio(
        "Choose Analysis Mode:",
        ["Single Review", "Batch Analysis (CSV)"]
    )
    
    # Single Review Mode
    if analysis_mode == "Single Review":
        st.header("🔍 Single Review Analysis")
        
        user_input = st.text_area(
            "Enter your review or feedback:",
            height=150,
            placeholder="Type or paste your review here..."
        )
        
        if st.button("Analyze", type="primary"):
            if user_input.strip():
                with st.spinner("Analyzing..."):
                    sentiment, polarity, color = analyze_sentiment(user_input)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sentiment", sentiment)
                    with col2:
                        st.metric("Polarity Score", f"{polarity:.2f}")
                    with col3:
                        st.metric("Confidence", f"{abs(polarity)*100:.1f}%")
                    
                    #Gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=polarity,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Sentiment Polarity"},
                        gauge={
                            'axis': {'range': [-1, 1]},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [-1, -0.1], 'color': "#7A3E3E"},
                                {'range': [-0.1, 0.1], 'color': "#A7A878"},
                                {'range': [0.1, 1], 'color': "#3E6B4A"}
                            ],
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    #Key Words
                    st.subheader("🔑 Key Topics/Keywords")
                    topics = extract_topics([user_input], n_topics=5)
                    st.write(", ".join(topics))
            else:
                st.warning("Please enter some text to analyze.")
    
    # Batch Analysis
    else:
        st.header("📁 Batch Analysis (CSV Upload)")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            st.subheader("Preview Data")
            st.dataframe(df.head(), use_container_width=True)
            st.write("")
            st.write("")
            text_column = st.selectbox(
                "Select the column containing reviews:",
                df.columns
            )
            st.write("")  
            st.write("")  
            if st.button("Analyze All Reviews", type="primary"):
                with st.spinner("Processing..."):
                    sentiments = []
                    polarities = []
                    
                    for text in df[text_column]:
                        if pd.notna(text):
                            sentiment, polarity, _ = analyze_sentiment(str(text))
                            sentiments.append(sentiment)
                            polarities.append(polarity)
                        else:
                            sentiments.append("Unknown")
                            polarities.append(0)
                    
                    df['Sentiment'] = sentiments
                    df['Polarity'] = polarities
                    
                    st.subheader("📊 Analysis Results")
                    col1, col2 = st.columns(2)
                    
                    #pie chart
                    with col1:
                        sentiment_counts = df['Sentiment'].value_counts()
                        fig_pie = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            title="Sentiment Distribution",
                            color_discrete_sequence=[
                                "#A1D99B",  
                                "#2A5470",  
                                "#009688"   
                            ]
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    #histogram
                    with col2:
                        fig_hist = px.histogram(
                            df,
                            x='Polarity',
                            title="Polarity Distribution",
                            nbins=30,
                            color_discrete_sequence=["#00C29A"]
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    #word cloud
                    st.subheader("☁️ Word Cloud")
                    all_text = ' '.join(df[text_column].dropna().astype(str))
                    wordcloud = generate_wordcloud(all_text)
                
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **How to use:**
        1. Choose Single Review or Batch Analysis
        2. Enter text or upload CSV file
        3. Click Analyze

        **Sentiment Scale:**
        - Positive: > 0.1
        - Neutral: -0.1 to 0.1
        - Negative: < -0.1
        """
    )

if __name__ == "__main__":
    main()
