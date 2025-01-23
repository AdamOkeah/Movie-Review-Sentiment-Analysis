import pandas as pd
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

#Loading the Dataset
file_path = "Train.csv"
data = pd.read_csv(file_path)
test = pd.read_csv("Test.csv")

# Limit the dataset to 10,000 rows for faster training
data = data.iloc[:10000]

#Preprocessing the Text
def preprocess_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)    # Remove mentions
    text = re.sub(r"#", "", text)       # Remove hashtags
    text = re.sub(r"[^\w\s]", "", text) # Remove special characters
    text = text.lower()                 # Convert to lowercase
    return text

data['cleaned_text'] = data['text'].apply(preprocess_text)
test['cleaned_text'] = test['text'].apply(preprocess_text)



# Step 3: Tokenise and Pad Sequences
max_vocab_size = 10000  # Maximum number of words in the vocabulary
max_seq_length = 100   # Maximum length of input sequences

tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(data['cleaned_text'])

# Tokenise and pad training data
train_sequences = tokenizer.texts_to_sequences(data['cleaned_text'])
X_train = pad_sequences(train_sequences, maxlen=max_seq_length, padding="post", truncating="post")
y_train = data['label'].values

# Tokenise and pad test data
test_sequences = tokenizer.texts_to_sequences(test['cleaned_text'])
X_test = pad_sequences(test_sequences, maxlen=max_seq_length, padding="post", truncating="post")
y_test = test['label'].values

# Step 4: Split Validation Data
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Step 5: Build the Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_vocab_size, output_dim=64, input_length=max_seq_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid")  # Binary classification
])

# Compilation and training of the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


print("Training the model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=32,
    verbose=1
)
#model evaluation
print("\nEvaluating the model...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

#Test the Model with Custom Inputs
def predict_sentiment(text):
    text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_seq_length, padding="post", truncating="post")
    prediction = model.predict(padded_sequence)[0][0]
    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    return sentiment

# Test with custom reviews
custom_reviews = [
    "The movie was fantastic and well-directed!",
    "I didn't like the film. It was too slow and boring.",
    "An amasing experience, I would watch it again.",
    "Terrible plot and poor acting."
]

print("\nCustom Predictions:")
for review in custom_reviews:
    print(f"Review: {review}\nPredicted Sentiment: {predict_sentiment(review)}\n")
