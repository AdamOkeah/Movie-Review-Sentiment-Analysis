# Sentiment Analysis of IMDB Movie Reviews

## Overview

This project implements a sentiment analysis model using TensorFlow to classify IMDB movie reviews as either "Positive" or "Negative". The model uses an LSTM-based neural network and processes textual data through tokenisation and padding. The dataset for training and testing was sourced from Kaggle.

---

## Dataset

The dataset used for this project is the **IMDB Dataset Sentiment Analysis in CSV format**, which can be found at the following link:  
[IMDB Dataset Sentiment Analysis](https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format)

### Dataset Structure:
- `Train.csv`: Contains the training data with two columns:
  - `text`: The movie review.
  - `label`: The sentiment label (`positive` or `negative`).
- `Test.csv`: Contains the test data with the same structure as `Train.csv`.

---

## Project Steps

###  **Preprocessing**
The text data is preprocessed to remove unwanted elements and standardise the format:
- URLs, mentions (`@username`), and hashtags (`#`) are removed.
- Special characters are stripped.
- All text is converted to lowercase.

### **Tokenisation and Padding**
- The text is tokenised using TensorFlow's `Tokenizer` to convert words into numerical representations.
- Padded sequences are generated to ensure all input data has the same length.

### **Model Architecture**
The model is a sequential neural network built with TensorFlow:
- **Embedding Layer**: For word vectorisation.
- **Bidirectional LSTM Layers**: To capture contextual information from both directions in the text.
- **Dense Layer**: For feature extraction with ReLU activation.
- **Dropout Layer**: To prevent overfitting.
- **Output Layer**: With sigmoid activation for binary classification.

### **Model Training**
The model is compiled with:
- **Loss Function**: Binary Crossentropy (suitable for binary classification).
- **Optimizer**: Adam.
- **Metrics**: Accuracy.

The model is trained on the `Train.csv` data and validated on a 20% split of the training data.

### **Evaluation**
The model is evaluated on the test dataset (`Test.csv`), and performance metrics such as accuracy and loss are reported.

### **Confusion Matrix**
A confusion matrix is generated to visually evaluate the model's performance on the test dataset, highlighting:
- True Positives
- True Negatives
- False Positives
- False Negatives

### **Custom Predictions**
The trained model is used to predict sentiment for custom movie reviews provided by the user.

---

## Requirements

The following libraries are required to run the project:
- Python 3.7 or later
- TensorFlow
- NumPy
- Pandas
- scikit-learn
- Matplotlib


## How to Run

1. Clone or download this repository.
2. Ensure the `Train.csv` and `Test.csv` files are placed in the same directory as the script.
3. Run the script:
   ```bash
   python sentiment_analysis.py
   ```

---

## Results

### Metrics:
- **Test Accuracy**: Displays after model evaluation.
- **Loss**: Displays the binary crossentropy loss.

### Visualisations:
- A **Confusion Matrix** is displayed to summarise the model's predictions.

### Custom Predictions:
The model makes predictions on user-provided reviews, such as:
```text
Review: The movie was fantastic and well-directed!
Predicted Sentiment: Positive
```

---

## Improvements

- Include hyperparameter tuning for better accuracy.
- Experiment with pre-trained embeddings like GloVe or Word2Vec.
- Extend the model for multi-class sentiment analysis (e.g., add "neutral" sentiment).

---

## Acknowledgements

- Dataset: [IMDB Dataset Sentiment Analysis in CSV format](https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format)
- TensorFlow: For the machine learning framework.
- Python Community: For open-source libraries used in this project.



