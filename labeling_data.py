import pandas as pd
from textblob import TextBlob
from tqdm.auto import tqdm  
from deep_translator import GoogleTranslator
import time  # Import the time module

def translate_text(text, lang='en', retries=3):
    for attempt in range(retries):
        try:
            if text:  
                translated = GoogleTranslator(source='auto', target=lang).translate(text)
                return translated
            else:
                return text
        except Exception as e:
            print(f"Error translating text: {e}. Retrying ({attempt + 1}/{retries})...")
            time.sleep(2)  
    return text  

def label_sentiment(text):
    """ Label the sentiment based on TextBlob analysis. """
    try:
        blob = TextBlob(text)
        if blob.sentiment.polarity > 0.1:
            return 'positif'
        elif blob.sentiment.polarity < -0.1:
            return 'negatif'
        else:
            return 'neutral'
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return 'neutral'

def main():
    df = pd.read_csv('cleaned_comments.csv', encoding='utf-8')
    
    tqdm.pandas(desc="Translating comments")
    df['English_Comment'] = df['Comment'].progress_apply(lambda x: translate_text(x))
    
    tqdm.pandas(desc="Labeling sentiments")
    df['Sentiment'] = df['English_Comment'].progress_apply(lambda x: label_sentiment(x))
    
    df.drop('English_Comment', axis=1, inplace=True)
    df.to_csv('labeled_comments.csv', index=False, encoding='utf-8')
    print("Data labeling complete. Labeled data saved to 'labeled_comments.csv'.")

if __name__ == "__main__":
    main()
