import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def load_data(filepath):
    
    return pd.read_csv(filepath)

def preprocess_data(df):
    X = df['Processed_Comment']  
    y = df['Sentiment']  
    return X, y

def feature_extraction(X):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_features = vectorizer.fit_transform(X)
    return X_features, vectorizer

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(random_state=42, class_weight='balanced')
    print("Training model...")
    model.fit(X_train, y_train)
    print("Model training complete.")
    y_pred = model.predict(X_test)
    return model, X_test, y_test, y_pred

def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

def plot_sentiment_distribution_by_category(df):
    sentiment_counts = df.groupby(['Category', 'Sentiment']).size().unstack(fill_value=0)
    sentiment_percentages = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0) * 100
    sentiment_percentages.plot(kind='bar', stacked=True)
    plt.ylabel('Percentage')
    plt.title('Percentage of Sentiments by Category')
    plt.legend(title='Sentiment')
    plt.show()

def main():
    filepath = 'labeled_comments.csv'
    df = load_data(filepath)
    print("Data loaded successfully.")
    X, y = preprocess_data(df)
    print("Data preprocessing complete.")
    X_features, vectorizer = feature_extraction(X)
    print("Feature extraction complete.")
    model, X_test, y_test, y_pred = train_model(X_features, y)
    evaluate_model(y_test, y_pred)
    
    df['Predicted_Sentiment'] = pd.Series(y_pred, index=y_test.index)
    plot_sentiment_distribution_by_category(df)

if __name__ == "__main__":
    main()
