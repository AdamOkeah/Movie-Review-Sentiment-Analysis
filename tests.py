import pandas as pd
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import IMBDsentanalysis

# Load the validation dataset
valid_file_path = "valid.csv"
valid_data = pd.read_csv(valid_file_path)

# Preprocess the text in the validation dataset
valid_data['cleaned_text'] = valid_data['text'].apply(preprocess_text)

# Tokenise and pad the validation data
valid_sequences = tokenizer.texts_to_sequences(valid_data['cleaned_text'])
X_valid = pad_sequences(valid_sequences, maxlen=max_seq_length, padding="post", truncating="post")
y_valid = valid_data['label'].values

# Load the saved model
model = tf.keras.models.load_model("RNN.h5")

# Predict on the validation data
y_valid_pred_probs = model.predict(X_valid)  # Predict probabilities
y_valid_pred = (y_valid_pred_probs >= 0.5).astype(int)  # Convert probabilities to binary labels

# Evaluate the model on the validation data
valid_loss, valid_accuracy = model.evaluate(X_valid, y_valid, verbose=1)
print(f"\nValidation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}")

# Generate a confusion matrix for the validation data
valid_cm = confusion_matrix(y_valid, y_valid_pred, labels=[0, 1])

# Visualise the confusion matrix
valid_disp = ConfusionMatrixDisplay(confusion_matrix=valid_cm, display_labels=["Negative", "Positive"])
valid_disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Validation Data")
plt.show()

#Predict sentiments for individual examples in the validation set
print("\nPredictions on Validation Data:")
for i, statement in enumerate(valid_data['statement'][:10]):  # Display the first 10 examples
    sentiment = predict_sentiment(statement)
    print(f"Statement: {statement}\nPredicted Sentiment: {sentiment}\n")
