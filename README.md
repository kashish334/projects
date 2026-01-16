# 🗨️Medical Q&A Chatbot

A simple and user-friendly medical question-answering chatbot built using the MedQuAD dataset from Kaggle.
The chatbot uses NLP and similarity search to provide medical information based on real datasets from trusted health sources.

This project is built with Streamlit and deployed on Streamlit Community Cloud.

Live app: [Medical Q&A Chatbot](https://madicalinfochat.streamlit.app/)

## Features
- Uses MedQuAD dataset for high-quality medical Q&A
- Clean, fast, interactive Streamlit interface
- Uses sentence embeddings + FAISS / ML models for similarity search
- Provides accurate and relevant medical answers
- Mobile-friendly and easy to use
- Fully deployed online on Streamlit Cloud

## Tech Stack
- Python, Streamlit for web UI.
- SentenceTransformer / embedding model for semantic search (or FAISS / sklearn similarity).
- TensorFlow / Keras model for intent or classification (if used).
- NLTK for text preprocessing and lemmatization.
- Google Generative AI / Gemini for answer summarization.

## Project Structure
- `app.py` – Main Streamlit app that loads the FAISS index, TF‑IDF vectorizer, label encoder, trained model and handles user chat.
- `training.ipynb` – Notebook used to preprocess data and train the chatbot model.
- `create_index.py` – Script to build the FAISS index and related artifacts from the QA dataset.
- `data/` – Input datasets (e.g., CSV/JSON with questions and answers).
- `chatbot_model.h5` – Trained Keras/TensorFlow model for intent/answer prediction.
- `tfidf_vectorizer.pkl` – Saved TF‑IDF vectorizer for converting text to vectors.
- `label_encoder.pkl` – Encodes and decodes class labels for answers/intents.
- `qa_answers.pkl` – Serialized mapping from labels/indices to final text answers.
- `faiss_index.bin` – Binary FAISS index file for fast semantic search.
- `requirements.txt` – Python dependencies for the project.
- `.devcontainer/` – Optional VS Code Dev Container configuration.

## Installation
- Clone the repository: `git clone https://github.com/kashish334/projects.git`
- Install dependencies: `pip install -r requirements.txt`
- Run the Streamlit app: `streamlit run app.py`

## Dataset
This project uses the MedQuAD (Medical Question Answering Dataset) from Kaggle.
It contains questions and answers collected from official medical sources like:
- NIH
- National Cancer Institute
- National Heart, Lung, and Blood Institute

## How It Works

- User question is preprocessed and converted into a TF‑IDF vector.
- FAISS is used to find similar questions in the dataset; their answers are retrieved.
- The trained neural network / classifier refines the prediction or selects the best answer.
- The final answer is displayed in the Streamlit UI, optionally with similarity score or related questions.

## Limitations

- Answers come from a fixed dataset and may not cover all medical conditions.
- This chatbot is for educational purposes only and must not be used as a substitute for professional medical advice.

