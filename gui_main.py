import streamlit as st
import pickle
import nltk
import faiss
import numpy as np
from nltk.stem import WordNetLemmatizer
import google.generativeai as genai
from sklearn.preprocessing import normalize
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import datetime
from dotenv import load_dotenv
load_dotenv()

for resource in ["punkt", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource)

lemmatizer = WordNetLemmatizer()
model = load_model("chatbot_model.h5")
labels = pickle.load(open("label_encoder.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

all_df = []
for main, subfolders, filename in os.walk("data"):
    for file in filename:
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(main, file))
            all_df.append(df)
data = pd.concat(all_df, ignore_index=True)

qa_questions = []
qa_answers = []

for index, row in data.iterrows():
    qa_questions.append(row['Question'])
    qa_answers.append(row['Answer'])

@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

@st.cache_data
def load_embeddings(qa_questions):
    embed_model = load_embed_model()
    return embed_model.encode(qa_questions,batch_size=32, show_progress_bar=False)

embed_model = load_embed_model()
qa_embeddings = np.array(load_embeddings(qa_questions)).astype('float32')
qa_embeddings = normalize(qa_embeddings)
qa_embeddings = np.array(qa_embeddings).astype('float32')
if len(qa_embeddings.shape) == 1:
    qa_embeddings = qa_embeddings.reshape(1, -1)
faiss.normalize_L2(qa_embeddings)

index = faiss.IndexFlatIP(qa_embeddings.shape[1])
index.add(qa_embeddings)

def semantic_search(user_question, top_k=3):
    query = load_embed_model().encode([user_question]).astype('float32')
    faiss.normalize_L2(query)
    distances, indices = index.search(query, k=top_k)
    return [qa_answers[i] for i in indices[0]]

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("Gemini API key not found! Check .env file.")
    st.stop()
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel("gemini-pro-latest")

def summarize_with_gemini(answer_text):
    prompt = (
        "Summarize the following medical explanation clearly and concisely:\n\n"
        f"{answer_text}"
    )
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Summarization failed: {e}"

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
        full_response = top_answers[0]
        summary = summarize_with_gemini(full_response)
        st.markdown(f"**🤖 Summary:** {summary}")
        with st.expander("📖 Full Answer"):
            st.write(full_response)

            
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