import os
import requests

FILES = {
    "chatbot_model.h5": "https://huggingface.co/Kashish-18/Medical-chatbot/resolve/main/chatbot_model.h5",
    "qa_answers.pkl": "https://huggingface.co/Kashish-18/Medical-chatbot/resolve/main/qa_answers.pkl",
    "label_encoder.pkl": "https://huggingface.co/Kashish-18/Medical-chatbot/resolve/main/label_encoder.pkl",
    "faiss_index.bin": "https://huggingface.co/Kashish-18/Medical-chatbot/resolve/main/faiss_index.bin",
    "tfidf_vectorizer.pkl" : "https://huggingface.co/Kashish-18/Medical-chatbot/resolve/main/tfidf_vectorizer.pkl"
}

def download_files():
    for file, url in FILES.items():
        if not os.path.exists(file):
            r = requests.get(url)
            r.raise_for_status()
            with open(file, "wb") as f:
                f.write(r.content)
