import streamlit as st
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision
from groq import Groq
from dotenv import load_dotenv

import uuid
import os

import tools  # your chunking file


# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()


# -----------------------------
# LOAD MODELS / CLIENTS
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_vector_db():
    return Endee()


@st.cache_resource
def load_llm():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


model = load_embedding_model()
client = load_vector_db()
llm = load_llm()


# -----------------------------
# SESSION STATE
# -----------------------------
if "index_name" not in st.session_state:
    st.session_state.index_name = f"rag_index_{uuid.uuid4().hex}"

if "index_created" not in st.session_state:
    st.session_state.index_created = False

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def build_prompt(query, contexts):
    context_text = "\n\n".join(contexts)

    return f"""
You are a helpful AI assistant.

Answer ONLY using the provided context.
If the answer is not in the context, say "I don't know".

Context:
{context_text}

Question:
{query}

Answer:
"""


def retrieve_context(query):
    index = client.get_index(st.session_state.index_name)

    query_embedding = model.encode([query])[0].tolist()

    results = index.query(
        query_embedding,
        top_k=5,
        ef=128,
        include_vectors=True,
    )

    contexts = []
    for item in results:
        if "meta" in item and "text" in item["meta"]:
            contexts.append(item["meta"]["text"])

    return contexts


def generate_answer(query, contexts):
    prompt = build_prompt(query, contexts)

    response = llm.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content


# -----------------------------
# UI
# -----------------------------
st.title("🧠 RAG Chatbot (Groq + Endee)")

# -----------------------------
# DOCUMENT INPUT
# -----------------------------
st.subheader("📄 Add Document")

uploaded_file = st.file_uploader("Upload a document (PDF or TXT)", type=["pdf", "txt"])

doc_text = st.text_area("OR paste text manually:")


if st.button("Process & Create Index"):
    final_text = ""

    # Priority: file > text
    if uploaded_file is not None:
        try:
            final_text = tools.parse_document(uploaded_file)
            st.success("✅ Document parsed successfully!")
        except Exception as e:
            st.error(f"Error parsing file: {e}")
            st.stop()

    elif doc_text.strip() != "":
        final_text = doc_text

    else:
        st.warning("Please upload a file or enter text")
        st.stop()

    # -----------------------------
    # SAME FLOW (UNCHANGED)
    # -----------------------------
    chunks = tools.chunk_text(final_text)
    embeddings = model.encode(chunks)

    if not st.session_state.index_created:
        client.create_index(
            name=st.session_state.index_name,
            dimension=384,
            space_type="cosine",
            precision=Precision.INT8,
        )
        st.session_state.index_created = True

    index = client.get_index(st.session_state.index_name)

    vectors = []
    for emb, chunk in zip(embeddings, chunks):
        vectors.append(
            {
                "id": str(uuid.uuid4()),
                "vector": emb.tolist(),
                "meta": {"text": chunk},
            }
        )

    index.upsert(vectors)

    st.success(f"✅ Added {len(vectors)} chunks!")


# -----------------------------
# CHAT SECTION
# -----------------------------
st.subheader("💬 Chat")

query = st.text_input("Ask something about your document:")

if st.button("Ask"):
    if not st.session_state.index_created:
        st.warning("⚠️ Please create index first!")
    elif query.strip() == "":
        st.warning("Enter a question")
    else:
        contexts = retrieve_context(query)
        answer = generate_answer(query, contexts)

        # Save chat
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Show sources
        with st.expander("📚 Retrieved Context"):
            for c in contexts:
                st.write("-", c)


# -----------------------------
# DISPLAY CHAT
# -----------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**🧑 You:** {msg['content']}")
    else:
        st.markdown(f"**🤖 Bot:** {msg['content']}")
