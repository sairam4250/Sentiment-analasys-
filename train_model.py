import os
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle

# Download required NLTK data
def download_nltk_data():
    print("Downloading NLTK data...")
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('movie_reviews', quiet=True)

def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Removing punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Removing numbers
    text = re.sub(r'\d+', '', text)
    # Tokenization
    tokens = text.split()
    # Removing stopwords and Lemmatization
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def main():
    download_nltk_data()
    
    os.makedirs('dataset', exist_ok=True)
    dataset_path = 'dataset/IMDB Dataset.csv'
    fallback_path = 'dataset/movie_reviews_sample.csv'
    
    if os.path.exists(dataset_path):
        print("Loading dataset from CSV...")
        df = pd.read_csv(dataset_path)
        # Using a subset for faster demonstration (optional)
        df = df.sample(min(10000, len(df)), random_state=42) 
        reviews = df['review'].tolist()
        sentiments = df['sentiment'].apply(lambda x: 1 if x.strip().lower() == 'positive' else 0).tolist()
    else:
        print("IMDB CSV not found. Loading NLTK 'movie_reviews' dataset as fallback...")
        from nltk.corpus import movie_reviews
        reviews = []
        sentiments = []
        for category in movie_reviews.categories():
            for fileid in movie_reviews.fileids(category):
                reviews.append(movie_reviews.raw(fileid))
                sentiments.append(1 if category == 'pos' else 0)
        
        # Save fallback dataset so the app can visualize it later
        df = pd.DataFrame({'review': reviews, 'sentiment': ['positive' if s == 1 else 'negative' for s in sentiments]})
        df.to_csv(fallback_path, index=False)
        print(f"Saved fallback dataset to {fallback_path}")
        
    print(f"Dataset size: {len(reviews)} reviews")
    
    print("Preprocessing text... (this may take a moment)")
    processed_reviews = [preprocess_text(review) for review in reviews]
    
    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(processed_reviews)
    y = np.array(sentiments)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC()
    }
    
    results = {}
    best_model = None
    best_accuracy = 0
    best_model_name = ""
    
    print("\nTraining and evaluating models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'Confusion Matrix': cm
        }
        
        print(f"[{name}] -> Accuracy: {acc:.4f} | F1: {f1:.4f}")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = name
            
    print(f"\nBest Model Selected: {best_model_name} with Accuracy: {best_accuracy:.4f}")
    
    # Save the best model and vectorizer
    os.makedirs('models', exist_ok=True)
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('models/results.pkl', 'wb') as f:
        pickle.dump(results, f)
        
    print("Models and evaluation metrics successfully saved to models/ directory.")

if __name__ == "__main__":
    main()
