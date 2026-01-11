import streamlit as st
import pickle
import nltk
import faiss
import numpy as np
from nltk.stem import WordNetLemmatizer
import google.generativeai as genai
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
    # Only loads the model once, keeping it in memory
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

@st.cache_resource
def load_faiss_components():
    """Loads the pre-calculated FAISS index and QA answers list."""
    try:
        # Load the saved index
        index = faiss.read_index("faiss_index.bin")
        
        # Load the saved QA Answers list
        with open("qa_answers.pkl", "rb") as f:
            qa_answers_loaded = pickle.load(f)
            
        return index, qa_answers_loaded
    
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e.filename}. Please run 'python create_index.py' first.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading FAISS components: {e}")
        st.stop()

# Load the components once at startup
index, qa_answers = load_faiss_components()
embed_model = load_embed_model()

def semantic_search(user_question, top_k=3):
    query = load_embed_model().encode([user_question]).astype('float32')
    faiss.normalize_L2(query)
    distances, indices = index.search(query, k=top_k)
    return [qa_answers[i] for i in indices[0]]

gemini_api_key = os.getenv("GEMINI_API_KEY")
st.write("API key loaded:", bool(api_key))
if not gemini_api_key:
    st.error("Gemini API key not found! Check .env file.")
    st.stop()
genai.configure(api_key=gemini_api_key)

model = genai.GenerativeModel("gemini-2.5-flash")

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

def truncate_answer(answer, max_sentences=2):
    sentences = answer.split(". ")
    if len(sentences) <= max_sentences:
        return answer
    return ". ".join(sentences[:max_sentences]) + "..."

# --- Define the Chat Processing Function ---
def process_user_input():
    # 1. Get the current user input from the key
    user_input = st.session_state.user_question_key
    
    # 2. Skip if input is empty
    if not user_input:
        return

    # 3. Process the question
    if user_input.lower() in ["exit", "quit", "bye"]:
        full_response = "Goodbye! Have a great day!"
    elif user_input.lower().startswith("search"):
        query = user_input[7:]
        full_response = f"You can search this on Google: https://www.google.com/search?q={query}"
    elif user_input.lower() == "time":
        full_response = f"The current time is {datetime.datetime.now().strftime('%H:%M:%S')}."
    else:
        top_answers = semantic_search(user_input, top_k=1)
        if top_answers:
            full_response = top_answers[0]
        else:
            full_response = "I'm not sure how to respond. Can you rephrase?"

    # 4. Summarize and Append to History
    # This part remains the same, but it is now inside the function
    summary = summarize_with_gemini(full_response)
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", {"summary": summary, "full": full_response}))

    # 5. Clear the input box after submission
    st.session_state.user_question_key = ""
    
st.set_page_config(page_title="Medical Q&A Chatbot", page_icon="💬")
st.title("💬 Medical Q&A Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []
if "user_question_key" not in st.session_state:
    st.session_state.user_question_key = "" # Initialize the key

# --- Update the Input Widget ---
# Add a key and the callback function
user_input = st.text_input(
    "Ask me a medical question:", 
    key="user_question_key", 
    on_change=process_user_input
)

for speaker, message in st.session_state.history:
    if speaker == "You":
        st.markdown(f"**🧑 You:** {message}")
    else:
        # message is {"summary": ..., "full": ...}
        st.markdown(f"**🤖 Summary:** {message['summary']}")
        with st.expander("📖 Full Answer"):
            st.write(message["full"])
            
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
