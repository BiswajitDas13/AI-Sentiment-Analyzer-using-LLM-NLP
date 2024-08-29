import streamlit as st
import openai
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize NLTK's VADER sentiment analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Function to analyze sentiment using GPT-4
def analyze_sentiment_gpt(text):
    openai.api_key = "your openai key"  # Ensure you set your OpenAI API key here
   
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Specify the model
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes sentiment."},
            {"role": "user", "content": f"Analyze the sentiment of the following text: '{text}'. Please provide the sentiment as Positive, Negative, or Neutral."}
        ],
        max_tokens=50
    )
    
    sentiment = response['choices'][0]['message']['content'].strip()
    return sentiment

# Function to analyze sentiment using VADER
def analyze_sentiment_vader(text):
    sentiment_score = sia.polarity_scores(text)
    if sentiment_score['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Streamlit UI
st.title("Sentiment Analyzer using LLM and NLP")

text_input = st.text_area("Enter text to analyze sentiment:")

if st.button("Analyze Sentiment"):
    if text_input:
        gpt_sentiment = analyze_sentiment_gpt(text_input)
        vader_sentiment = analyze_sentiment_vader(text_input)
        
        st.write("**Sentiment Analysis using GPT-4:**", gpt_sentiment)
        st.write("**Sentiment Analysis using VADER (NLTK):**", vader_sentiment)
    else:
        st.warning("Please enter some text to analyze.")
