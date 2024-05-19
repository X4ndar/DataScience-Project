import os
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# Define Arabic and Moroccan Darija stop words
arabic_darija_stop_words = set([
    "من", "في", "على", "و", "فى", "لم", "ما", "كما", "هذا", "أن", "هو", "عن", "هذه",
    "به", "كان", "إلى", "التي", "الذي", "الذين", "أنا", "علي", "أو", "إذا", "أي", "هل",
    "لكن", "عند", "أنت", "كل", "نحن", "هم", "مع", "يا", "بين", "إلى", "بها", "بس", "واش",
    "فين", "علاش", "اش", "كيف", "هادي", "هاد", "هاذا", "واحد", "ايه", "لا", "لي", "معاك", "ديال"
])

def normalize_arabic(text):
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'[ى]', 'ي', text)
    text = re.sub(r'[ؤ]', 'و', text)
    text = re.sub(r'[ئ]', 'ي', text)
    text = re.sub(r'[ة]', 'ه', text)
    return text

def clean_text(text):
    # Normalize text
    text = normalize_arabic(text)
    # Remove non-word characters and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Tokenize and remove stop words
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in arabic_darija_stop_words]
    return ' '.join(tokens)

def load_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc="Loading data"):
            parts = line.strip().split(', ')
            if len(parts) == 4:
                category, article, author, comment = parts
                data.append({
                    "Category": category.split(": ")[1],
                    "Article": article.split(": ")[1],
                    "Author": author.split(": ")[1],
                    "Comment": comment.split(": ")[1]
                })
    return pd.DataFrame(data)

def clean_data(df):
    tqdm.pandas(desc="Cleaning data")
    # Apply custom text cleaning
    df['Processed_Comment'] = df['Comment'].progress_apply(clean_text)
    df = df.drop_duplicates()
    df = df[df['Processed_Comment'].str.strip() != '']
    return df

def main():
    filepath = 'comments.txt'  
    df = load_data(filepath)
    df = clean_data(df)
    print(df.head())

    # Define a safe path to save the file
    save_path = os.path.join(os.getcwd(), 'cleaned_comments.csv')
    try:
        df.to_csv(save_path, index=False)
        print(f"Data cleaning complete. Cleaned data saved to '{save_path}'.")
    except PermissionError:
        print("Permission denied: Unable to write to the file. Please check your file permissions or close the file if it is open in another program.")

if __name__ == "__main__":
    main()
