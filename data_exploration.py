import pandas as pd
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import normalize_alef_maksura_ar, normalize_alef_ar
from tqdm import tqdm

# Manually define Arabic stop words
arabic_stop_words = set([
    "من", "في", "على", "و", "فى", "يا", "مع", "أن", "هو", "عن", "هذا",
    "به", "لم", "بين", "إلى", "أنا", "ذلك", "علي", "أو", "لي", "كان",
    "كما", "له", "إذا", "هذه", "ما", "كل", "تكون", "هل", "هم", "نحن",
    "لو", "فيه", "التي", "مثل", "هذا", "أم", "بها", "لها", "منه", "الذي",
    "معه", "عليه", "أنت", "إلي", "لا", "منذ", "هنا", "هناك", "الى",
    "اللذين", "ألا", "بل", "أيضا", "حتى"
])

# Function to load data from a CSV file
def load_data(filepath):
    return pd.read_csv(filepath)

# Function to process text by normalizing, tokenizing, and removing stop words
def process_text(text):
    # Normalize text to ensure uniform character representation
    text = normalize_alef_ar(normalize_alef_maksura_ar(dediac_ar(text)))
    # Tokenize the text into individual words
    tokens = simple_word_tokenize(text)
    # Filter out stop words from the tokens
    filtered_tokens = [token for token in tokens if token not in arabic_stop_words]
    # Join the tokens back into a single string
    return ' '.join(filtered_tokens)

# Main function to orchestrate data loading, processing, and saving
def main():
    filepath = 'cleaned_comments.csv'  # Path to the CSV file containing the comments
    df = load_data(filepath)
    
    # Apply text processing to the 'Comment' column and save the result to a new column
    tqdm.pandas(desc="Processing comments")
    df['Processed_Comment'] = df['Comment'].progress_apply(process_text)
    
    # Print the first few rows of the DataFrame to verify the processed comments
    print(df[['Author', 'Processed_Comment']].head())
    # Save the DataFrame with the processed comments back to a CSV file
    df.to_csv('processed_comments.csv', index=False)
    print("Text processing complete. Data saved to 'processed_comments.csv'.")

if __name__ == "__main__":
    main()
