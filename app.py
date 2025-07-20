import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

# Load the trained model
model_path = r"D:\ML_AI_projects\NLP\Bank_Pulse_application\sentiment_dashboard\sentiment_model.pkl"
model = joblib.load(model_path)

# Colors
sentiment_colors = {
    "Positive": "#2ca02c",  # Green
    "Neutral": "#ff7f0e",   # Orange
    "Negative": "#d62728"   # Red
}

# Title
st.title("📊Bank Pulse : Sentiment Analysis Dashboard")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

    st.write("### Sample Data")
    st.write(data.head())

    # Predictions
    reviews = data['Review_Text']
    predictions = model.predict(reviews)
    data['Sentiment'] = predictions

    # Categorization
    issue_keywords = {
        "Service-related": ["service", "response", "downtime", "helpline", "communication", "support", "helpdesk", "issue", "delay"],
        "Branch-related": ["branch", "branches", "location", "atm", "cash machine", "teller", "queue", "line", "waiting", "counter"],
        "Staff-related": ["staff", "representative", "employee", "manager", "team", "teller", "officer", "agent"],
        "Misbehavior": ["rude", "unhelpful", "misbehave", "incompetent", "argue", "aggressive", "impolite", "insult"],
        "Charges": ["charges", "fee", "fees", "penalty", "interest"],
        "Software-Related": ["downtime", "bug", "crash", "error", "login", "update", "app", "mobile", "website"],
        "ATM-related": ["atm", "cash machine", "atm’s", "atm,", "atm."]
    }

    def categorize_review(text):
        text_lower = str(text).lower()
        scores = {}
        for cat, kws in issue_keywords.items():
            scores[cat] = sum(bool(re.search(rf"\b{k}\b", text_lower)) for k in kws)
        best_cat, best_score = max(scores.items(), key=lambda x: x[1])
        return best_cat if best_score > 0 else "Other"

    data["Issue_Category"] = data["Review_Text"].apply(categorize_review)

    # Date parsing
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

    # --- Sidebar Filters ---
    st.sidebar.title("📅 Filter Data")

    # Date filter - default shows the entire period
    start_date = data['Date'].min().date()
    end_date = data['Date'].max().date()

    # Set the default to Custom Period
    selected_period = st.sidebar.radio("Select Period", options=["Custom Period"], index=0)

    # Custom period selection
    start_date = st.sidebar.date_input("Start Date", start_date)
    end_date = st.sidebar.date_input("End Date", end_date)

    # Filter data based on the selected period
    filtered_data = data[
        (data['Date'] >= pd.to_datetime(start_date)) & 
        (data['Date'] <= pd.to_datetime(end_date))
    ]

    # Display filtered data
    st.write("### Filtered Data")
    st.write(filtered_data)

    # Predictions
    reviews = filtered_data['Review_Text']
    predictions = model.predict(reviews)
    filtered_data['Sentiment'] = predictions

    # --- Trend Chart ---
    st.subheader("📈 Review Trend Over Time")

    filtered_data['Month'] = filtered_data['Date'].dt.to_period('M')
    trend_data = filtered_data.groupby(['Month', 'Sentiment']).size().unstack(fill_value=0)

    fig1, ax1 = plt.subplots(figsize=(12,6))
    for sentiment in ["Positive", "Neutral", "Negative"]:
        if sentiment in trend_data.columns:
            ax1.plot(trend_data.index.to_timestamp(), trend_data[sentiment], marker='o', label=sentiment, color=sentiment_colors[sentiment])

    ax1.set_xlabel('Month')
    ax1.set_ylabel('Number of Reviews')
    ax1.set_title(f'Monthly Trend by Sentiment ({selected_period} - {start_date} to {end_date})')
    ax1.legend(title='Sentiment')
    ax1.grid(True)

    st.pyplot(fig1)

    # --- Pie Chart ---
    sentiment_counts = filtered_data['Sentiment'].value_counts()

    fig2, ax2 = plt.subplots()
    colors = [sentiment_colors.get(s, 'grey') for s in sentiment_counts.index]
    ax2.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
    ax2.axis('equal')
    st.pyplot(fig2)

    # --- Stacked Bar Chart ---
    category_sentiment = filtered_data.groupby(['Issue_Category', 'Sentiment']).size().unstack(fill_value=0)

    fig3, ax3 = plt.subplots(figsize=(12,6))

    bottom = None
    for sentiment in ["Positive", "Neutral", "Negative"]:
        if sentiment in category_sentiment.columns:
            ax3.bar(category_sentiment.index, category_sentiment[sentiment], 
                    label=sentiment, 
                    bottom=bottom, 
                    color=sentiment_colors[sentiment])
            if bottom is None:
                bottom = category_sentiment[sentiment]
            else:
                bottom += category_sentiment[sentiment]

    plt.xticks(rotation=45, ha='right')
    ax3.set_xlabel('Issue Category')
    ax3.set_ylabel('Number of Reviews')
    ax3.set_title('Stacked Issue Category Distribution by Sentiment')
    ax3.legend()
    st.pyplot(fig3)

    # --- WordCloud ---
    all_text = " ".join(filtered_data['Review_Text'].dropna().astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

    st.subheader("🌟 Word Cloud of Reviews")
    fig4, ax4 = plt.subplots(figsize=(10,5))
    ax4.imshow(wordcloud, interpolation='bilinear')
    ax4.axis('off')
    st.pyplot(fig4)

    # --- Average Review Length by Sentiment ---
    st.subheader("📝 Average Review Length by Sentiment")
    filtered_data['Review_Length'] = filtered_data['Review_Text'].apply(lambda x: len(str(x).split()))

    avg_review_length = filtered_data.groupby('Sentiment')['Review_Length'].mean().reset_index()

    fig7, ax7 = plt.subplots(figsize=(8,5))
    sns.barplot(x='Sentiment', y='Review_Length', data=avg_review_length, palette=[sentiment_colors['Positive'], sentiment_colors['Neutral'], sentiment_colors['Negative']])
    ax7.set_title('Average Review Length by Sentiment')
    ax7.set_ylabel('Average Length (Words)')
    st.pyplot(fig7)

    # --- Heatmap of Issue Category vs Sentiment ---
    st.subheader("🌡️ Heatmap of Issue Category vs Sentiment")
    category_sentiment_matrix = filtered_data.groupby(['Issue_Category', 'Sentiment']).size().unstack(fill_value=0)

    fig9, ax9 = plt.subplots(figsize=(10,6))
    sns.heatmap(category_sentiment_matrix, annot=True, fmt='g', cmap="YlGnBu", cbar=True, ax=ax9)
    ax9.set_title("Issue Category vs Sentiment Heatmap")
    st.pyplot(fig9)

    # --- Download Filtered Data ---
    st.subheader("📥 Download Filtered Data")
    csv = filtered_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"filtered_reviews_{start_date}_{end_date}.csv",
        mime='text/csv'
    )

else:
    st.warning("Please upload a CSV file to get started.")
