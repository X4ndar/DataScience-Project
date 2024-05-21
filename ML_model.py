import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    X = df['Processed_Comment']  
    y = df['Sentiment']  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, X_test

def feature_extraction(X_train, X_test):
    """ Use TF-IDF to convert text data into numerical data with bi-grams included. """
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    return X_train_features, X_test_features

def train_model(X_train, y_train, X_test, test_comments):
    """ Train a Logistic Regression classifier with class weight balancing. """
    model = LogisticRegression(random_state=42, class_weight='balanced')
    print("Training model...")
    model.fit(X_train, y_train)
    print("Model training complete.")
    y_pred = model.predict(X_test)
    r = pd.DataFrame({"comment": test_comments, "predicted Sentiment": y_pred})
    print(r.head(50))
    return y_pred

def evaluate_model(y_test, y_pred):
    """ Evaluate the model performance with accuracy and classification report. """
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
   # print(classification_report(y_test, y_pred))

def main():
    filepath = 'labeled_comments.csv'
    df = load_data(filepath)
    print("Data loaded successfully.")
    X_train, X_test, y_train, y_test, test_comments = preprocess_data(df)
    print("Data preprocessing complete.")
    X_train_features, X_test_features = feature_extraction(X_train, X_test)
    print("Feature extraction complete.")
    y_pred = train_model(X_train_features, y_train, X_test_features, test_comments)
    evaluate_model(y_test, y_pred)

if __name__ == "__main__":
    main()
