import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io

# Import functions
from model import (
    preprocess_text,
    analyze_sentiment,
    extract_topics,
    generate_wordcloud,
)

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Sentiment & Topic Classifier",
    page_icon="📊",
    layout="wide"
)

# ==========================================================
# CSS CODE
# ==========================================================

custom_css = """
<style>

/* Main page background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(
        135deg,
        #001419,
        #003B46,
        #005F73
    ) !important;

    background-attachment: fixed;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(
        135deg,
        #001419,
        #00323D
    ) !important;
}

/* Text */
* {
    color: #E8FDF9 !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(
        90deg,
        #007F8C,
        #00C6A2
    ) !important;

    color: white !important;
    border: none !important;

    padding: 0.6rem 1.2rem !important;

    font-weight: 600 !important;

    border-radius: 10px !important;

    transition: 0.3s !important;
}

/* Hover */
.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #00C6A2,
        #007F8C
    ) !important;

    transform: scale(1.02);
}

</style>
"""

st.markdown(
    custom_css,
    unsafe_allow_html=True
)


# ==========================================================
# MAIN FUNCTION
# ==========================================================

def main():

    # ======================================================
    # TITLE
    # ======================================================

    st.title("📈 ReviewXAI")

    st.markdown(
        """
        <p style='font-size: 22px;'>
        Classify Reviews using AI-powered sentiment analysis
        to identify positive, negative and neutral opinions
        </p>
        """,
        unsafe_allow_html=True
    )

    # ======================================================
    # SIDEBAR OPTIONS
    # ======================================================

    st.sidebar.header("⚙️ Options")

    analysis_mode = st.sidebar.radio(
        "Choose Analysis Mode:",
        [
            "Single Review",
            "Batch Analysis (CSV)"
        ]
    )

    # ======================================================
    # SINGLE REVIEW MODE
    # ======================================================

    if analysis_mode == "Single Review":

        st.header("🔍 Single Review Analysis")

        user_input = st.text_area(
            "Enter the Review you want to analyse:",
            height=150,
            placeholder="Type or paste your review here..."
        )

        # ==================================================
        # ANALYZE BUTTON
        # ==================================================

        if st.button(
            "Analyze",
            type="primary"
        ):

            if user_input.strip():

                with st.spinner("Analyzing..."):

                    # Sentiment analysis
                    sentiment, polarity, color = analyze_sentiment(
                        user_input
                    )

                    # ==================================================
                    # METRICS
                    # ==================================================

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Sentiment",
                            sentiment
                        )

                    with col2:

                        st.metric(
                            "Polarity Score",
                            f"{polarity:.2f}"
                        )

                    with col3:

                        st.metric(
                            "Confidence",
                            f"{abs(polarity) * 100:.1f}%"
                        )

                    # ==================================================
                    # GAUGE
                    # ==================================================

                    fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",

                            value=polarity,

                            domain={
                                "x": [0, 1],
                                "y": [0, 1]
                            },

                            title={
                                "text": "Sentiment Polarity"
                            },

                            gauge={

                                "axis": {
                                    "range": [-1, 1]
                                },

                                "bar": {
                                    "color": color
                                },

                                "steps": [

                                    {
                                        "range": [-1, -0.1],
                                        "color": "#7A3E3E"
                                    },

                                    {
                                        "range": [-0.1, 0.1],
                                        "color": "#A7A878"
                                    },

                                    {
                                        "range": [0.1, 1],
                                        "color": "#3E6B4A"
                                    }

                                ]
                            }
                        )
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    # ==================================================
                    # KEY TOPICS
                    # ==================================================

                    st.subheader(
                        "🔑 Key Topics/Keywords"
                    )

                    topics = extract_topics(
                        [user_input],
                        n_topics=5
                    )

                    if topics:

                        st.write(
                            ", ".join(topics)
                        )

                    else:

                        st.info(
                            "No topics could be extracted."
                        )

            else:

                st.warning(
                    "Please enter some text to analyze."
                )

    # ==========================================================
    # BATCH ANALYSIS MODE
    # ==========================================================

    else:

        st.header(
            "📁 Batch Analysis (CSV Upload)"
        )

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"]
        )

        # ======================================================
        # FILE UPLOADED
        # ======================================================

        if uploaded_file is not None:

            # ==================================================
            # ROBUST CSV READING
            # ==================================================

            try:

                # Read uploaded file as bytes
                file_bytes = uploaded_file.getvalue()

                # Possible encodings
                encodings = [
                    "utf-8",
                    "utf-8-sig",
                    "cp1252",
                    "latin1",
                    "ISO-8859-1"
                ]

                df = None

                # ------------------------------------------------
                # TRY NORMAL CSV PARSER
                # ------------------------------------------------

                for encoding in encodings:

                    try:

                        df = pd.read_csv(
                            io.BytesIO(file_bytes),
                            encoding=encoding
                        )

                        break

                    except (
                        UnicodeDecodeError,
                        pd.errors.ParserError
                    ):

                        continue

                # ------------------------------------------------
                # TRY PYTHON ENGINE
                # ------------------------------------------------

                if df is None:

                    for encoding in encodings:

                        try:

                            df = pd.read_csv(
                                io.BytesIO(file_bytes),
                                encoding=encoding,
                                engine="python",
                                on_bad_lines="skip"
                            )

                            break

                        except Exception:

                            continue

                # ------------------------------------------------
                # IF FILE STILL CANNOT BE READ
                # ------------------------------------------------

                if df is None:

                    st.error(
                        """
                        Unable to read this CSV file.

                        Please make sure that the uploaded file is
                        a valid CSV file with properly formatted rows.
                        """
                    )

                    st.stop()

            except Exception as e:

                st.error(
                    f"Error reading CSV file: {str(e)}"
                )

                st.stop()

            # ==================================================
            # CHECK IF DATAFRAME IS EMPTY
            # ==================================================

            if df.empty:

                st.error(
                    "The uploaded CSV file is empty."
                )

                st.stop()

            # ==================================================
            # PREVIEW DATA
            # ==================================================

            st.subheader(
                "👀 Preview Data"
            )

            st.dataframe(
                df.head(),
                use_container_width=True
            )

            st.write("")

            # ==================================================
            # DATASET INFORMATION
            # ==================================================

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Rows",
                    df.shape[0]
                )

            with col2:

                st.metric(
                    "Columns",
                    df.shape[1]
                )

            with col3:

                st.metric(
                    "Missing Values",
                    int(df.isnull().sum().sum())
                )

            st.write("")

            # ==================================================
            # SELECT TEXT COLUMN
            # ==================================================

            text_column = st.selectbox(
                "Select the column containing reviews/tweets:",
                df.columns
            )

            st.write("")

            # ==================================================
            # ANALYZE ALL REVIEWS
            # ==================================================

            if st.button(
                "Analyze All Reviews",
                type="primary"
            ):

                with st.spinner(
                    "Processing reviews..."
                ):

                    sentiments = []
                    polarities = []

                    # ==================================================
                    # PROCESS EACH REVIEW
                    # ==================================================

                    for text_value in df[text_column]:

                        if pd.notna(text_value):

                            try:

                                sentiment, polarity, _ = (
                                    analyze_sentiment(
                                        str(text_value)
                                    )
                                )

                                sentiments.append(
                                    sentiment
                                )

                                polarities.append(
                                    polarity
                                )

                            except Exception:

                                sentiments.append(
                                    "Unknown"
                                )

                                polarities.append(
                                    0
                                )

                        else:

                            sentiments.append(
                                "Unknown"
                            )

                            polarities.append(
                                0
                            )

                    # ==================================================
                    # ADD RESULTS TO DATAFRAME
                    # ==================================================

                    df["Sentiment"] = sentiments

                    df["Polarity"] = polarities

                    # ==================================================
                    # RESULTS
                    # ==================================================

                    st.subheader(
                        "📊 Analysis Results"
                    )

                    # ==================================================
                    # SENTIMENT COUNTS
                    # ==================================================

                    sentiment_counts = (
                        df["Sentiment"]
                        .value_counts()
                    )

                    # ==================================================
                    # CHARTS
                    # ==================================================

                    col1, col2 = st.columns(2)

                    # ------------------------------------------------
                    # PIE CHART
                    # ------------------------------------------------

                    with col1:

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

                        st.plotly_chart(
                            fig_pie,
                            use_container_width=True
                        )

                    # ------------------------------------------------
                    # HISTOGRAM
                    # ------------------------------------------------

                    with col2:

                        fig_hist = px.histogram(

                            df,

                            x="Polarity",

                            title="Polarity Distribution",

                            nbins=30,

                            color_discrete_sequence=[
                                "#00C29A"
                            ]
                        )

                        st.plotly_chart(
                            fig_hist,
                            use_container_width=True
                        )

                    # ==================================================
                    # SENTIMENT SUMMARY
                    # ==================================================

                    st.subheader(
                        "📌 Sentiment Summary"
                    )

                    summary_col1, summary_col2, summary_col3 = (
                        st.columns(3)
                    )

                    positive_count = (
                        df["Sentiment"]
                        .astype(str)
                        .str.lower()
                        .eq("positive")
                        .sum()
                    )

                    negative_count = (
                        df["Sentiment"]
                        .astype(str)
                        .str.lower()
                        .eq("negative")
                        .sum()
                    )

                    neutral_count = (
                        df["Sentiment"]
                        .astype(str)
                        .str.lower()
                        .eq("neutral")
                        .sum()
                    )

                    with summary_col1:

                        st.metric(
                            "😊 Positive",
                            positive_count
                        )

                    with summary_col2:

                        st.metric(
                            "😐 Neutral",
                            neutral_count
                        )

                    with summary_col3:

                        st.metric(
                            "😞 Negative",
                            negative_count
                        )

                    # ==================================================
                    # WORD CLOUD
                    # ==================================================

                    st.subheader(
                        "☁️ Word Cloud"
                    )

                    try:

                        # Collect all valid reviews
                        text_data = (
                            df[text_column]
                            .dropna()
                            .astype(str)
                            .tolist()
                        )

                        if len(text_data) > 0:

                            # Combine reviews
                            all_text = " ".join(
                                text_data
                            )

                            # Generate word cloud
                            wordcloud_result = (
                                generate_wordcloud(
                                    all_text
                                )
                            )

                            if wordcloud_result is not None:

                                st.image(
                                    wordcloud_result,
                                    use_container_width=True
                                )

                            else:

                                st.info(
                                    "Word cloud could not be generated."
                                )

                        else:

                            st.info(
                                "No text available for word cloud."
                            )

                    except Exception as e:

                        st.warning(
                            "Word cloud could not be generated."
                        )

                    # ==================================================
                    # CLASSIFIED DATA
                    # ==================================================

                    st.subheader(
                        "📋 Classified Reviews"
                    )

                    st.dataframe(
                        df,
                        use_container_width=True
                    )

                    # ==================================================
                    # DOWNLOAD RESULTS
                    # ==================================================

                    st.subheader(
                        "⬇️ Download Results"
                    )

                    result_csv = df.to_csv(
                        index=False,
                        encoding="utf-8"
                    )

                    st.download_button(
                        label="⬇️ Download Results CSV",

                        data=result_csv,

                        file_name="ReviewXAI_Results.csv",

                        mime="text/csv"
                    )

    # ==========================================================
    # SIDEBAR INFORMATION
    # ==========================================================

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


# ==========================================================
# RUN APPLICATION
# ==========================================================

if __name__ == "__main__":
    main()
