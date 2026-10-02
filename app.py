import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Vehicle Damage & Fraud Assistant",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Vehicle Damage & Fraud Assistant")
st.write(
    "Ask questions about vehicle damage, repair information, "
    "and fraud-related records."
)


# ============================================================
# 2. LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("vehicle_rag_documents.csv")

    return df


df = load_data()


# ============================================================
# 3. LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    return model


embedding_model = load_embedding_model()


# ============================================================
# 4. CREATE FAISS INDEX
# ============================================================

@st.cache_resource
def create_faiss_index():

    texts = df["text"].tolist()

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    # Normalize embeddings
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


index = create_faiss_index()


# ============================================================
# 5. GROQ CLIENT
# ============================================================

# Put your Groq API key in Streamlit Secrets.
#
# Example:
#
# GROQ_API_KEY = "your_api_key"
#
# Do NOT put your real API key directly in GitHub.

groq_api_key = st.secrets.get("GROQ_API_KEY")

if not groq_api_key:

    st.error(
        "Groq API key is missing. "
        "Please add GROQ_API_KEY to Streamlit Secrets."
    )

    st.stop()


client = Groq(
    api_key=groq_api_key
)


# ============================================================
# 6. RETRIEVE RELEVANT DOCUMENTS
# ============================================================

def retrieve_documents(question, top_k=5):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    faiss.normalize_L2(question_embedding)

    scores, indices = index.search(
        question_embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx != -1:

            results.append({
                "text": df.iloc[idx]["text"],
                "score": float(score)
            })

    return results


# ============================================================
# 7. GENERATE ANSWER WITH GROQ
# ============================================================

def generate_answer(question, retrieved_documents):

    context = "\n\n".join(
        [
            doc["text"]
            for doc in retrieved_documents
        ]
    )

    prompt = f"""
You are a Vehicle Damage and Fraud Information Assistant.

Answer the user's question using the provided context.

Do not invent information.

If the answer cannot be found in the context,
clearly say that the available records do not contain
enough information to answer the question.

Context:
{context}

User Question:
{question}

Answer:
"""

    response = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions using retrieved "
                    "vehicle-related information."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2
    )

    return response.choices[0].message.content


# ============================================================
# 8. CHAT INTERFACE
# ============================================================

question = st.text_input(
    "Ask your question:",
    placeholder="Example: What type of vehicle damage is shown?"
)


if question:

    with st.spinner("Searching vehicle records..."):

        documents = retrieve_documents(
            question,
            top_k=5
        )

    with st.spinner("Generating answer..."):

        answer = generate_answer(
            question,
            documents
        )

    st.subheader("Answer")

    st.write(answer)


    # ========================================================
    # 9. SHOW RETRIEVED INFORMATION
    # ========================================================

    with st.expander("View Retrieved Records"):

        for i, document in enumerate(documents):

            st.markdown(
                f"### Retrieved Record {i + 1}"
            )

            st.write(
                document["text"]
            )

            st.write(
                f"Similarity Score: "
                f"{document['score']:.4f}"
            )

            st.divider()


# ============================================================
# 10. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About")

    st.write(
        "This application uses:"
    )

    st.write("• Hugging Face embeddings")
    st.write("• FAISS vector search")
    st.write("• Groq LLM")
    st.write("• Retrieval-Augmented Generation (RAG)")

    st.write(
        f"Records available: {len(df)}"
    )
