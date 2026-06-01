import streamlit as st
import requests
import time


st.set_page_config(
    page_title="Production RAG",
    page_icon="🤖",
    layout="wide"
)


if "messages" not in st.session_state:

    st.session_state.messages = []


with st.sidebar:

    st.title("RAG Settings")

    chunk_type = st.selectbox(
        "Chunking",
        [
            "recursive",
            "fixed",
            "markdown",
            "semantic",
            "parent-child"
        ]
    )

    embedding_type = st.selectbox(
        "Embedding",
        [
            "bge",
            "instructor",
            "e5",
            "nomic"
        ]
    )

    retrieval_type = st.selectbox(
        "Retrieval",
        [
            "hybrid",
            "semantic",
            "mmr",
            "multi-query",
            "parent-child"
        ]
    )

    reranker_type = st.selectbox(
        "Reranker",
        [
            "bge"
        ]
    )

    generation_model = st.selectbox(
        "Generation Model",
        [
            "ollama"
        ]
    )

    k = st.slider(
        "Top K",
        min_value=1,
        max_value=20,
        value=5
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1
    )


st.markdown(
    """
    <h1 style='text-align: center;'>
        Production RAG Chatbot
    </h1>
    """,
    unsafe_allow_html=True
)

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

query = st.chat_input("Ask your question...")

if query:

    
    with st.chat_message("user"):

        st.markdown(query)

    
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    payload = {

        "session_id": "user_1",

        "query": query,

        "chunk_type": chunk_type,
        "embedding_type": embedding_type,
        "retrieval_type": retrieval_type,
        "reranker_type": reranker_type,
        "generation_model": generation_model,

        "k": k,
        "temperature": temperature
    }

    try:

        start = time.time()

        response = requests.post(
            "http://127.0.0.1:8000/chat",
            json=payload,
            timeout=300
        )

        end = time.time()

        latency = round(end - start, 2)

        if response.status_code != 200:

            st.error(f"API Error: {response.text}")

        else:

            data = response.json()

            assistant_response = data["response"]

            with st.chat_message("assistant"):

                st.markdown(assistant_response)

                st.divider()

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Latency",
                        f"{latency} sec"
                    )

                with col2:
                    st.metric(
                        "Retrieved Docs",
                        len(data["retrieved_docs"])
                    )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": assistant_response
                }
            )

    except requests.exceptions.Timeout:

        st.error(
            "Request timed out. Your RAG pipeline is taking too long."
        )

    except Exception as e:

        st.error(f"Error: {str(e)}")