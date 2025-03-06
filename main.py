"""
Text Generation Project - Complete implementation in one file
This file contains all necessary components for data collection, preprocessing,
model building, training, evaluation, and Streamlit deployment.
"""

import os
import re
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import requests
from bs4 import BeautifulSoup
import streamlit as st

import nltk
nltk.download('punkt')

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')

# Directory setup
os.makedirs('data', exist_ok=True)


#######################
# 1. DATA COLLECTION #
#######################

def download_gutenberg_text(url, output_file):
    """Download text data from Project Gutenberg"""
    response = requests.get(url)
    if response.status_code == 200:
        # Find the start and end of the actual text content
        text = response.text

        # Basic cleanup - this might need adjustment based on specific books
        start_marker = "*** START OF"
        end_marker = "*** END OF"

        if start_marker in text and end_marker in text:
            start_idx = text.find(start_marker)
            start_idx = text.find("\n", start_idx) + 1
            end_idx = text.find(end_marker)

            # Extract the text between markers
            extracted_text = text[start_idx:end_idx].strip()

            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(extracted_text)

            print(f"Text saved to {output_file}")
            return True
        else:
            print("Could not find text markers in the document")
            return False
    else:
        print(f"Failed to download: {response.status_code}")
        return False


def scrape_text_data():
    """
    Function to scrape text data from websites
    Can be extended to include more sources
    """
    # Example: Download a book from Project Gutenberg
    urls = [
        "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
        "https://www.gutenberg.org/files/84/84-0.txt",  # Frankenstein
        "https://www.gutenberg.org/files/1661/1661-0.txt"  # The Adventures of Sherlock Holmes
    ]

    # Download and compile corpus
    compiled_text = ""
    for i, url in enumerate(urls):
        output_file = f"data/book_{i + 1}.txt"
        if download_gutenberg_text(url, output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                compiled_text += f.read() + "\n\n"

    # Save the compiled corpus
    with open("data/corpus.txt", 'w', encoding='utf-8') as f:
        f.write(compiled_text)

    return "data/corpus.txt"


#######################
# 2. PREPROCESSING #
#######################

def preprocess_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters and numbers (optional)
    text = re.sub(r'[^\w\s]', '', text)

    # Simple tokenization by splitting on spaces instead of using word_tokenize
    tokens = text.split()

    return ' '.join(tokens)


def prepare_training_data(file_path, sequence_length=50):
    """Prepare training data for the model"""
    # Load text
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Preprocess
    processed_text = preprocess_text(text)

    # Tokenize
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([processed_text])
    total_words = len(tokenizer.word_index) + 1

    # Create input sequences
    input_sequences = []
    tokens = processed_text.split()

    for i in range(len(tokens) - sequence_length):
        seq = tokens[i:i + sequence_length + 1]
        input_sequences.append(seq)

    # Pad and create training data
    X = []
    y = []

    for seq in input_sequences:
        X.append(seq[:-1])
        y.append(seq[-1])

    X_seq = tokenizer.texts_to_sequences(X)
    X_pad = pad_sequences(X_seq, maxlen=sequence_length)

    # Convert y to integers but don't convert to one-hot encoding
    y_integers = []
    for word in y:
        word_seq = tokenizer.texts_to_sequences([[word]])[0]
        if word_seq:
            y_integers.append(word_seq[0])
        else:
            y_integers.append(0)

    y_integers = np.array(y_integers)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_pad, y_integers, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, tokenizer, total_words


#######################
# 3. MODEL BUILDING #
#######################

def build_model(total_words, sequence_length=50, embedding_dim=100, rnn_units=256):
    """Build the text generation model"""
    model = Sequential([
        Embedding(total_words, embedding_dim, input_length=sequence_length),
        LSTM(rnn_units, return_sequences=True),
        Dropout(0.2),
        LSTM(rnn_units),
        Dropout(0.2),
        Dense(total_words, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',  # Changed from categorical_crossentropy
        metrics=['accuracy']
    )

    return model


#######################
# 4. TRAINING #
#######################

def train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=64):
    """Train the model"""
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Save the model
    model.save('text_generation_model.h5')

    return history


#######################
# 5. TEXT GENERATION #
#######################

def generate_text(model, tokenizer, seed_text, next_words=50, sequence_length=50, temperature=1.0):
    """Generate text using the trained model"""
    generated_text = seed_text

    for _ in range(next_words):
        # Tokenize and pad the input sequence
        token_list = tokenizer.texts_to_sequences([generated_text.split()[-sequence_length:]])
        token_list = pad_sequences(token_list, maxlen=sequence_length)

        # Predict the next word
        predictions = model.predict(token_list, verbose=0)[0]

        # Apply temperature for creativity control
        predictions = np.log(predictions) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)

        # Sample from the predictions
        predicted_index = np.random.choice(len(predictions), p=predictions)

        # Convert index to word
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break

        # Add to generated text
        generated_text += " " + output_word

    return generated_text


#######################
# 6. EVALUATION #
#######################

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Calculate perplexity
    perplexity = np.exp(loss)
    print(f"Perplexity: {perplexity:.4f}")

    return loss, accuracy, perplexity


#######################
# 7. STREAMLIT APP #
#######################

def run_streamlit_app():
    """Run the Streamlit application"""
    st.title("Text Generation App")
    st.write("This application generates creative text based on your prompt")

    # Input for seed text
    seed_text = st.text_area("Enter a seed text:", "Once upon a time")

    # Slider for number of words to generate
    next_words = st.slider("Number of words to generate:", 10, 200, 50)

    # Slider for temperature (creativity)
    temperature = st.slider("Temperature (higher = more creative):", 0.2, 2.0, 1.0)

    # Button to generate text
    if st.button("Generate Text"):
        try:
            # Load the model and tokenizer
            model = load_model('text_generation_model.h5')

            # For simplicity in this example, we'll just retokenize the data
            # In a production app, you'd save and load the tokenizer
            with open("data/corpus.txt", 'r', encoding='utf-8') as f:
                text = f.read()
            processed_text = preprocess_text(text)
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts([processed_text])

            # Generate text
            generated_text = generate_text(
                model, tokenizer, seed_text,
                next_words=next_words,
                temperature=temperature
            )

            # Display the generated text
            st.write("### Generated Text:")
            st.write(generated_text)

        except Exception as e:
            st.error(f"Error generating text: {str(e)}")
            st.info("Make sure you've trained the model first by running the main() function")


#######################
# 8. MAIN FUNCTION #
#######################

def main():
    """Main function to run the entire pipeline"""
    print("Starting Text Generation Project")

    # 1. Data Collection
    print("Step 1: Collecting data...")
    corpus_path = scrape_text_data()

    # 2. Data Preprocessing
    print("Step 2: Preprocessing data...")
    X_train, X_test, y_train, y_test, tokenizer, total_words = prepare_training_data(corpus_path)
    print(f"Total words in vocabulary: {total_words}")
    print(f"Training data shape: {X_train.shape}")

    # 3. Model Building
    print("Step 3: Building model...")
    model = build_model(total_words)
    model.summary()

    # 4. Training
    print("Step 4: Training model...")
    train_model(model, X_train, y_train, X_test, y_test, epochs=10)

    # 5. Evaluation
    print("Step 5: Evaluating model...")
    loss, accuracy, perplexity = evaluate_model(model, X_test, y_test)

    # 6. Text Generation Example
    print("Step 6: Generating sample text...")
    seed_text = "once upon a time"
    generated_text = generate_text(model, tokenizer, seed_text)
    print(f"Seed: '{seed_text}'")
    print(f"Generated: '{generated_text}'")

    # Save tokenizer words (as a simple way to persist the tokenizer)
    with open('tokenizer_words.txt', 'w', encoding='utf-8') as f:
        for word, index in tokenizer.word_index.items():
            f.write(f"{word},{index}\n")

    print("\nText generation model is ready!")
    print("Run the Streamlit app to interact with it: streamlit run this_file.py")


#######################
# 9. ENTRY POINT #
#######################

if __name__ == "__main__":
    # If you want to run the streamlit app directly
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'app':
        run_streamlit_app()
    else:
        # Otherwise run the full pipeline
        main()