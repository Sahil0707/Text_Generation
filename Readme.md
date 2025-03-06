# Text Generation Project
A Python-based text generation system using LSTM neural networks trained on classic literature, with a Streamlit web interface for interactive text generation.

# Overview
This project implements a generative model for text data capable of producing creative and grammatically correct sentences based on input prompts. The model is trained on a dataset of classic literature from Project Gutenberg and uses recurrent neural networks (LSTM) for sequence prediction.

# Features

-Data Collection: Automatically downloads classic literature texts from Project Gutenberg
-Text Preprocessing: Cleans and processes text for machine learning
-LSTM Neural Network: Implements a deep learning architecture for text generation
-Temperature Control: Adjusts creativity vs. predictability of generated text
-Web Interface: Streamlit-based UI for easy interaction with the model
-Performance Metrics: Evaluates model using accuracy and perplexity

# Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/Sahil0707/Text_Generation.git

2. Create a virtual environment:
   ```sh
   python -m venv .venv
   source .venv/bin/activate  # Mac/Linux
   .venv\Scripts\activate     # Windows

3. Install dependencies:
   ```sh
   pip install -r requirements.txt

4. Download required NLTK data:
   ```sh
   import nltk
   nltk.download('punkt')
# Train the Model
Run the main script to collect data, train the model, and perform evaluation:

```sh
python main.py

