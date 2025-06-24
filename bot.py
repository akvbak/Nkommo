import os
import api
import gradio as gr

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from ghana_nlp import GhanaNLP
import voice

# --- Configuration ---
os.environ["OPENAI_API_KEY"] = api.APIKEY
os.environ["GHANA_NLP_API_KEY"] = voice.APIKEY

# Paths
datapath = r"Nkommo/files"
chromapath = r"Nkommo/chroma_db"

# Embeddings & Model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.5)
nlp = GhanaNLP(voice.APIKEY)

# Chroma vectorstore
vector_store = Chroma(
    collection_name="Nkommov1",
    embedding_function=embeddings_model,
    persist_directory=chromapath,
)
retriever = vector_store.as_retriever(search_kwargs={'k': 5})

# --- GhanaNLP Helpers ---
def transcribe_audio(filepath: str) -> str:
    response = nlp.stt(filepath)
    
    if isinstance(response, str):
        return response
    elif isinstance(response, dict) and "text" in response:
        return response["text"]
    else:
        raise ValueError("Unexpected response format from nlp.stt()")

    

def translate_text(text, target_lang="tw-en"):
    response = nlp.translate(text, target_lang)

    if isinstance(response, str):
        return response
    
    elif isinstance(response, dict):
        if "translation" in response:
            return response["translation"]
        elif "message" in response:
            print(f"[WARN] Translation failed: {response['message']}")
            return text  # fallback: return original input
        else:
            raise ValueError(f"Unknown dictionary format: {response}")

    return text  # fallback: return original input


# --- Core Stream Response ---
def stream_response(input_data, history, is_audio=False, source_lang="twi"):
    """
    Accept either a file path (if is_audio=True) or raw text message.
    Transcribes & translates audio, then runs RAG via LangChain + Chroma + OpenAI.
    Yields updated (chatbot, state) tuples for Gradio streaming.
    """
    # Ensure history is initialized as list of tuples
    history = history or []

    # Map user selection to API language codes
    lang_map = {
        "twi": "tw",
        "ga": "ga",
        "ewe": "ee",
        "hausa": "ha",
        "english": "en"
    }
    source_code = lang_map.get(source_lang.lower(), "tw")
    lang_pair = f"{source_code}-en"

    # Prepare the user message text
    if is_audio and isinstance(input_data, str):
        # Using GhanaNLP for transcription and translation
        raw_text = transcribe_audio(input_data)
        print(f"[Transcribed] {raw_text}")
        user_message = translate_text(raw_text, target_lang=lang_pair)
        print(f"[Translated] {user_message}")
    else:
        user_message = input_data

    # Append user turn
    history.append((user_message, ""))

    # Retrieve relevant docs and build knowledge
    docs = retriever.invoke(user_message)
    knowledge = "".join([doc.page_content + "\n\n" for doc in docs])

    # Build RAG prompt
    rag_prompt = f"""
You are an assistant which answers questions based on knowledge which is provided to you.
While answering, you can use your own intuition and discretion only if you cannot provide
an answer based on the information in the \"The knowledge\" section.
Do not make it known that you are using the knowledge provided to you or that you were trained on any knowledge at all.
Do not repeat the question in your answer. Answer in a concise and clear manner.

The question: {user_message}

Conversation history: {history}

The knowledge: {knowledge}
"""

    # Stream response chunks
    accumulated = ""
    for chunk in llm.stream(rag_prompt):
        accumulated += chunk.content
        # Update bot turn in history
        history[-1] = (user_message, accumulated)
        # Yield both chatbot and state
        yield history, history

