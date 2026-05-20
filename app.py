import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
import os

# Download NLTK data if not present
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

download_nltk_resources()

# Page Configuration
st.set_page_config(page_title="Movie Review Sentiment Analysis", page_icon="🎬", layout="wide")

# Custom CSS for Dark Mode Aesthetics
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .success-text {
        color: #00FF00;
        font-size: 26px;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        background-color: rgba(0, 255, 0, 0.1);
        border-left: 5px solid #00FF00;
    }
    .error-text {
        color: #FF4B4B;
        font-size: 26px;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        background-color: rgba(255, 75, 75, 0.1);
        border-left: 5px solid #FF4B4B;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 16px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    model_path = 'models/best_model.pkl'
    vectorizer_path = 'models/tfidf_vectorizer.pkl'
    results_path = 'models/results.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return None, None, None
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
        
    return model, vectorizer, results

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def main():
    st.title("🎬 Sentiment Analysis on Movie Reviews")
    st.markdown("Predict whether a movie or product review is **Positive** or **Negative** using Machine Learning.")
    
    model, vectorizer, results = load_models()
    
    if model is None or vectorizer is None:
        st.error("⚠️ Models not found! Please run `python train_model.py` first to train and save the models.")
        return

    # Create Tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Predict Sentiment", "📊 Model Metrics", "📈 Visualizations"])
    
    with tab1:
        st.subheader("Real-time Prediction")
        
        sample_reviews = [
            "This movie was an absolute masterpiece! The acting was phenomenal and the plot was gripping.",
            "Terrible film. I wasted two hours of my life. The storyline was completely nonsensical and boring.",
            "It was an average experience. Had some good action sequences but the pacing felt a bit off.",
            "The product exceeded my expectations. Built quality is superb and it arrived early."
        ]
        
        selected_sample = st.selectbox("Choose a sample review (or write your own below):", ["-- Custom Review --"] + sample_reviews)
        
        user_input = st.text_area("Enter Review:", value=selected_sample if selected_sample != "-- Custom Review --" else "", height=150)
        
        if st.button("Predict Sentiment 🚀"):
            if user_input.strip() == "":
                st.warning("Please enter a review to analyze.")
            else:
                with st.spinner('Analyzing sentiment...'):
                    processed_input = preprocess_text(user_input)
                    vectorized_input = vectorizer.transform([processed_input])
                    
                    prediction = model.predict(vectorized_input)[0]
                    
                    confidence = None
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(vectorized_input)[0]
                        confidence = max(proba) * 100
                    elif hasattr(model, "decision_function"):
                        decision = model.decision_function(vectorized_input)[0]
                        import math
                        prob = 1 / (1 + math.exp(-decision))
                        confidence = max(prob, 1 - prob) * 100
                    
                    st.markdown("### Result:")
                    if prediction == 1:
                        st.markdown('<div class="success-text">🟢 Positive Sentiment</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="error-text">🔴 Negative Sentiment</div>', unsafe_allow_html=True)
                    
                    if confidence:
                        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

    with tab2:
        st.subheader("Model Evaluation Metrics")
        if results:
            df_results = pd.DataFrame(results).T.drop(columns=['Confusion Matrix'])
            
            st.write("### Algorithm Comparison")
            st.dataframe(df_results.style.highlight_max(axis=0, color='#1f77b4'), use_container_width=True)
            
            st.write("### Accuracy Chart")
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            sns.barplot(x=df_results.index, y=df_results['Accuracy'], palette='coolwarm', ax=ax)
            plt.ylim(0, 1.0)
            plt.xticks(color='white')
            plt.yticks(color='white')
            plt.xlabel("Models", color='white')
            plt.ylabel("Accuracy", color='white')
            st.pyplot(fig)
            
            st.write("### Confusion Matrix (Best Model)")
            best_model_name = df_results['Accuracy'].idxmax()
            cm = results[best_model_name]['Confusion Matrix']
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            fig2.patch.set_facecolor('#0e1117')
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            st.pyplot(fig2)

    with tab3:
        st.subheader("Dataset Visualizations")
        dataset_path = 'dataset/IMDB Dataset.csv'
        fallback_path = 'dataset/movie_reviews_sample.csv'
        
        df = None
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path).sample(min(2000, sum(1 for line in open(dataset_path)) - 1), random_state=42)
        elif os.path.exists(fallback_path):
            df = pd.read_csv(fallback_path)
            
        if df is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Sentiment Distribution")
                fig3, ax3 = plt.subplots()
                fig3.patch.set_facecolor('#0e1117')
                ax3.set_facecolor('#0e1117')
                sns.countplot(x='sentiment', data=df, palette='Set2', ax=ax3)
                plt.xticks(color='white')
                plt.yticks(color='white')
                plt.xlabel("Sentiment", color='white')
                plt.ylabel("Count", color='white')
                st.pyplot(fig3)
                
            with col2:
                st.write("### Word Cloud (Positive Reviews)")
                try:
                    pos_reviews = ' '.join(df[df['sentiment'].str.lower().str.strip() == 'positive']['review'].dropna().tolist())
                    if pos_reviews:
                        wordcloud = WordCloud(width=800, height=400, background_color='#0e1117', max_words=100, colormap='Greens').generate(pos_reviews)
                        fig4, ax4 = plt.subplots(figsize=(10, 5))
                        fig4.patch.set_facecolor('#0e1117')
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis('off')
                        st.pyplot(fig4)
                    else:
                        st.write("No positive reviews found to generate word cloud.")
                except Exception as e:
                    st.error(f"Could not generate word cloud: {e}")
        else:
            st.info("Dataset not found for visualization. Run training first to generate a sample dataset.")

if __name__ == "__main__":
    main()
