import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# For converting text to numerical data
from sklearn.feature_extraction.text import TfidfVectorizer 

# For training a machine learning model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK data (if not already installed)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

nltk.download()
nltk.download('stopwords')
nltk.download('punkt')

# Example dataset
data = {
    "Review": [
        "The movie was fantastic, love it!",
        "Horrible movie, waste of time.",
        "It was an okay film, not great but not terrible also.",
        "Absolutely brilliant. A must-watch!",
        "Terrible acting and predictable plot."
    ],
    "Sentiment": ["Positive", "Negative", "Neutral", "Positive", "Negative"]
}

df = pd.DataFrame(data)

# 1. Preprocess the data
# Text preprocessing function
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    tokens = word_tokenize(text)  # Tokenize
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(tokens)

df['Cleaned_Review'] = df['Review'].apply(preprocess_text)

# Display the cleaned dataset
print(df[['Review', 'Cleaned_Review']])

# 2. Visualize the data
sns.countplot(x='Sentiment', data=df)
plt.title('Sentiment Distribution')
plt.show()

# 3. Convert to numerical data
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['Cleaned_Review']).toarray()
y = df['Sentiment']

# 4. Train a machine learning model
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 5. Test the model
new_reviews = ["I hated this movie.", "What an amazing experience!", "The plot was dull and boring."]
new_reviews_cleaned = [preprocess_text(review) for review in new_reviews]
new_reviews_vectorized = vectorizer.transform(new_reviews_cleaned).toarray()

predictions = model.predict(new_reviews_vectorized)
for review, sentiment in zip(new_reviews, predictions):
    print(f"Review: {review} => Sentiment: {sentiment}")
