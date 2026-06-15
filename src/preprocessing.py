import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import re
# pyrefly: ignore [missing-import]
import nltk
# pyrefly: ignore [missing-import]
from nltk.corpus import stopwords
# pyrefly: ignore [missing-import]
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data (run once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text):
        """
        Clean text data:
        - Convert to lowercase
        - Remove special characters and punctuation
        - Remove stopwords
        - Lemmatization (Convert words to their base form)
        """
        if not isinstance(text, str):
            return ""
            
        # 1. Lowercase
        text = text.lower()
        
        # 2. Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # 3. Tokenize (simple split)
        tokens = text.split()
        
        # 4. Remove stopwords and lemmatize
        cleaned_tokens = [
            self.lemmatizer.lemmatize(word) 
            for word in tokens 
            if word not in self.stop_words
        ]
        
        return ' '.join(cleaned_tokens)

def load_and_preprocess_data(file_path):
    """
    Read and preprocess the entire dataset.
    """
    print(f"Reading data from {file_path}...")
    # Use pd.read_excel if it's an Excel file, otherwise default to csv
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    
    # Find the column containing the requirement text
    possible_cols = ['RequirementText', 'Requirement']
    text_col = next((col for col in possible_cols if col in df.columns), None)
    if not text_col:
        raise ValueError(f"Text column not found. Please ensure the file contains one of these columns: {possible_cols}")
    
    print("Cleaning text data...")
    preprocessor = TextPreprocessor()
    df['CleanedText'] = df[text_col].apply(preprocessor.clean_text)
    
    # Remove rows with empty text after cleaning
    df = df[df['CleanedText'].str.strip().astype(bool)]
    
    print("Preprocessing completed!")
    return df

def get_features_and_labels(df, text_col='CleanedText', max_features=5000):
    """
    Extract features using TF-IDF and separate labels.
    """
    print("Converting text to Vectors (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df[text_col])
    
    # Label columns are the remaining columns (Multi-label classification problem)
    ignore_cols = ['ProjectID', 'RequirementText', 'Requirement', 'CleanedText', 'Dataset_Name']
    label_cols = [c for c in df.columns if c not in ignore_cols]
    
    y = df[label_cols]
    
    return X, y, vectorizer
