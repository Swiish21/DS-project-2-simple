# README: Sentiment Analysis Project

## Project Overview
This project focuses on sentiment analysis of text data using natural language processing (NLP) and machine learning techniques. The goal is to classify movie reviews as **Positive**, **Negative**, or **Neutral** based on their textual content.

## Features
- **Data Preprocessing**: Clean and preprocess text data, including:
  - Removing special characters and extra spaces.
  - Converting text to lowercase.
  - Tokenizing text and removing stopwords.
- **Visualization**: Display the sentiment distribution using a bar chart.
- **Text Vectorization**: Convert text into numerical representations using TF-IDF.
- **Model Training**: Train a logistic regression model on the dataset.
- **Prediction**: Classify new movie reviews into sentiment categories.

---

## Requirements
### Python Libraries
Install the following Python libraries before running the project:
- `pandas`
- `seaborn`
- `matplotlib`
- `scikit-learn`
- `nltk`

### Installing Dependencies
Run the following command to install the required libraries:
```bash
pip install pandas seaborn matplotlib scikit-learn nltk
```

---

## Dataset
### Example Dataset
The dataset used in this project contains movie reviews and their corresponding sentiments. Below is a sample of the data:
| Review                                          | Sentiment  |
|------------------------------------------------|------------|
| The movie was fantastic, love it!             | Positive   |
| Horrible movie, waste of time.                | Negative   |
| It was an okay film, not great but not terrible.| Neutral    |
| Absolutely brilliant. A must-watch!           | Positive   |
| Terrible acting and predictable plot.         | Negative   |

---

## How to Run
1. **Clone or Download the Project**: 
   Save the code to your local machine.

2. **Run the Code**:
   Execute the script in a Python environment (e.g., Jupyter Notebook, VS Code, or a Python terminal).

3. **Input New Reviews**:
   Add new reviews to the `new_reviews` list in the script to test the model's predictions.

---

## Key Steps
### 1. Preprocessing
- Text is cleaned by removing special characters, converting to lowercase, and tokenizing.
- Stopwords are removed to retain meaningful words.

### 2. Visualization
- A bar chart shows the distribution of sentiments in the dataset.

### 3. Vectorization
- Text data is converted into numerical form using **TF-IDF Vectorization** with a limit of 1000 features.

### 4. Model Training
- A **Logistic Regression** model is trained to classify the sentiment of reviews.

### 5. Testing
- New reviews are preprocessed and vectorized before being classified by the trained model.

---

## Results
The performance of the model is evaluated using:
- **Accuracy**: Measures how often the model's predictions are correct.
- **Classification Report**: Provides precision, recall, and F1-score for each sentiment class.

---

## Example Predictions
Below are sample predictions made by the model:
| Review                             | Predicted Sentiment |
|------------------------------------|---------------------|
| I hated this movie.                | Negative            |
| What an amazing experience!        | Positive            |
| The plot was dull and boring.      | Negative            |

---

## Future Improvements
- Add more data to improve model accuracy.
- Experiment with other machine learning models or deep learning techniques.
- Use more advanced preprocessing techniques, like stemming or lemmatization.

---

## Contact
If you have any questions or feedback, feel free to reach out.