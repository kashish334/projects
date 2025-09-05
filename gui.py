import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import webbrowser
import datetime

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()
model = load_model("chatbot_model.h5")
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

def parse_xml_intents(data_folder="data"):
        import os
        import xml.etree.ElementTree as ET
        all_intents = []
        for foldername, subfolders, filenames in os.walk(data_folder):
            for filename in filenames:
                if filename.endswith(".xml"):
                    file_path = os.path.join(foldername, filename)
                    tree = ET.parse(file_path)
                    root = tree.getroot()

                    tag_elem = root.find("Focus")
                    
                    if tag_elem is None or tag_elem.text is None:
                        print(f"⚠️  Skipping {file_path}: <Focus> tag missing or empty.")
                        continue
                    tag = tag_elem.text.strip()

                    # Get all questions from <QAPairs>
                    qa_pairs = root.find("QAPairs")
                    if qa_pairs is not None:
                        patterns = []
                        responses = []
                        for qapair in qa_pairs.findall("QAPair"):
                            question = qapair.find("Question")
                            answer = qapair.find("Answer")
                            if question is not None and question.text:
                                patterns.append(question.text.strip())
                                if answer is not None and answer.text:
                                    responses.append(answer.text.strip())

                        # Only add if patterns found
                        if patterns and responses:
                            intent = {
                                "tag": tag,
                                "patterns": patterns,
                                "responses": responses,
                            }
                            all_intents.append(intent)

        print(f"Parsed {len(all_intents)} intents from XML")

        return all_intents

intents = parse_xml_intents()
# Flatten intents into Q&A pairs
qa_questions = []
qa_answers = []

for intent in intents:
    for q, a in zip(intent["patterns"], intent["responses"]):
        qa_questions.append(q)
        qa_answers.append(a)

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode all questions once
qa_embeddings = embed_model.encode(qa_questions)

def semantic_search(user_question, top_k=1):
    user_emb = embed_model.encode([user_question])
    similarities = cosine_similarity(user_emb, qa_embeddings)[0]
    
    # Get top-k similar answers
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [qa_answers[i] for i in top_indices]


st.set_page_config(page_title="Medical Q&A Chatbot", page_icon="💬")
st.title("💬 Medical Q&A Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Ask me a medical question:")

def truncate_answer(answer, max_sentences=2):
    sentences = answer.split(". ")
    if len(sentences) <= max_sentences:
        return answer
    return ". ".join(sentences[:max_sentences]) + "..."

if user_input:
    if user_input.lower() in ["exit", "quit", "bye"]:
        response = "Goodbye! Have a great day!"
        full_response = response
    elif user_input.lower().startswith("search"):
        query = user_input[7:]
        response = f"You can search this on Google: https://www.google.com/search?q={query}"
        full_response = response
    elif user_input.lower() == "time":
        response = f"The current time is {datetime.datetime.now().strftime('%H:%M:%S')}."
        full_response = response
    else:
        top_answers = semantic_search(user_input, top_k=1)
        if top_answers:
            full_response = top_answers[0]
            response = truncate_answer(full_response, max_sentences=2)
        else:
            response = "I'm not sure how to respond. Can you rephrase?"
            full_response = response

    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", (response, full_response)))

for speaker, message in st.session_state.history:
    if speaker == "You":
        st.markdown(f"**🧑 You:** {message}")
    else:
        short_resp, full_resp = message
        st.markdown(f"**🤖 Bot:** {short_resp}")
        with st.expander("📖 Read full answer"):
            st.write(full_resp)
            
col1, col2 = st.columns(2)

with col1:
    if st.button("🧹 Clear Chat"):
        st.session_state.history = []

with col2:
    if st.button("💾 Save Chat"):
        if st.session_state.history:
            chat_text = ""
            for speaker, msg in st.session_state.history:
                chat_text += f"{speaker}: {msg}\n"
            filename = f"chat_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            st.download_button("📥 Download Chat History", chat_text, file_name=filename)
        else:
            st.warning("Chat is empty!")
