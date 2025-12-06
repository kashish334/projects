import faiss
import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# --- 1. Load Data (Same as your app) ---
all_df = []
for main, subfolders, filename in os.walk("data"):
    for file in filename:
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(main, file))
            all_df.append(df)
data = pd.concat(all_df, ignore_index=True)

qa_questions = [row['Question'] for index, row in data.iterrows()]
qa_answers = [row['Answer'] for index, row in data.iterrows()]

# --- 2. Load Embedder Model ---
# This is a one-time load for the creation script
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu") 

# --- 3. Create Embeddings & Index ---
print("Starting embedding creation... this may take a few minutes.")
qa_embeddings = embed_model.encode(qa_questions, batch_size=32, show_progress_bar=True).astype('float32')

# Normalize
qa_embeddings = normalize(qa_embeddings)
if len(qa_embeddings.shape) == 1:
    qa_embeddings = qa_embeddings.reshape(1, -1)
faiss.normalize_L2(qa_embeddings)
qa_embeddings = np.array(qa_embeddings).astype('float32')

# Create Index
index = faiss.IndexFlatIP(qa_embeddings.shape[1])
index.add(qa_embeddings)
print("Embedding and indexing complete.")

# --- 4. SAVE THE INDEX and Embeddings ---
if index.ntotal > 0:
    try:
        # Save FAISS Index
        faiss.write_index(index, "faiss_index.bin")
        print("✅ FAISS index successfully saved to faiss_index.bin")
        
        # Save QA Answers list using pickle (Crucial for look-up!)
        import pickle
        with open("qa_answers.pkl", "wb") as f:
            pickle.dump(qa_answers, f)
        print("✅ QA answers list saved to qa_answers.pkl")
        
    except Exception as e:
        print(f"❌ ERROR saving files: {e}")
else:
    print("❌ Index is empty. Check your data.")