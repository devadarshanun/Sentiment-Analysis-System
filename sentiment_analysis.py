import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Step 1: Load Dataset
df = pd.read_csv('sentiment_dataset.csv')
df = pd.read_csv('sentiment_dataset.csv')
print("Dataset Loaded Successfully!\n")
print(df.head())  # Shows first 5 rows
print(f"Total rows loaded: {len(df)}")


# Step 2: Preprocessing Function
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()  # Simple tokenization
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Apply preprocessing
df['Clean_Text'] = df['Text'].apply(preprocess)

# Step 3: Feature Extraction
# No split, use full dataset
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Clean_Text'])
y = df['Sentiment']

model = LogisticRegression()
model.fit(X, y)

# Predict on same data
y_pred = model.predict(X)

print("Model Evaluation:\n")
print("Accuracy:", accuracy_score(y, y_pred))
print("\nClassification Report:\n", classification_report(y, y_pred))

# Step 7: User Input Prediction
print("\n--- Sentiment Prediction System ---")
while True:
    user_input = input("\nEnter text (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    processed = preprocess(user_input)
    features = vectorizer.transform([processed])
    prediction = model.predict(features)
    print(f"Predicted Sentiment: {prediction[0]}")
