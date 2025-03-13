# Install required libraries
!pip install scikit-learn nltk

# Import necessary libraries

import re
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Define a small dataset manually
documents = [

    "buy cheap medicines online",  # Spam
    "online pharmacy offers discount",  # Spam
    "meeting scheduled with team at 3 PM",  # Not Spam
    "project update and presentation tomorrow",  # Not Spam
    "exclusive offer on medicines and health",  # Spam
]
labels = [1, 1, 0, 0, 1]  # 1 = Spam, 0 = Not Spam

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Apply preprocessing
processed_documents = [preprocess_text(doc) for doc in documents]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(processed_documents, labels, test_size=0.2, random_state=42)
# Create a text classification pipeline (TF-IDF + Naïve Bayes)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Predictions on test data
y_pred = model.predict(X_test)

# Evaluate the model
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Test with new example inputs
test_sentences = [
    "discount on medicines available",
    "team meeting tomorrow at 10 AM",
    "pharmacy offers cheap medicines",
    "project report submission today",
    "online pharmacy exclusive deal"
]

# Predict and display results
predictions = model.predict(test_sentences)
for sentence, label in zip(test_sentences, predictions):
    print(f"'{sentence}' -> {'Spam' if label == 1 else 'Not Spam'}")

