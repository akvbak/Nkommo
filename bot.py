import os
import api
import voice
import asyncio
import functools

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from ghana_nlp import GhanaNLP

# --- Configuration ---
os.environ["OPENAI_API_KEY"] = api.APIKEY
os.environ["GHANA_NLP_API_KEY"] = voice.APIKEY

# Paths
DATAPATH = r"Nkommo/files"
CHROMAPATH = r"Nkommo/chroma_db"

# Models
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.5, streaming=True)
nlp = GhanaNLP(voice.APIKEY)

# Vector store (loaded, not rebuilt)
vector_store = Chroma(
    collection_name="Nkommov1",
    embedding_function=embeddings_model,
    persist_directory=CHROMAPATH,
)
retriever = vector_store.as_retriever(search_kwargs={'k': 3})

# --- GhanaNLP Helpers ---
def transcribe_audio(filepath: str) -> str:
    response = nlp.stt(filepath)
    if isinstance(response, str):
        return response
    elif isinstance(response, dict) and "text" in response:
        return response["text"]
    print("[DEBUG] Raw STT response:", response)
    raise ValueError("Unexpected response format from nlp.stt()")

@functools.lru_cache(maxsize=256)
def translate_text(text, target_lang="tw-en"):
    response = nlp.translate(text, target_lang)
    if isinstance(response, str):
        return response
    elif isinstance(response, dict):
        if "translation" in response:
            return response["translation"]
        elif "message" in response:
            print(f"[WARN] Translation failed: {response['message']}")
            return text
        raise ValueError(f"Unknown dictionary format: {response}")
    return text

# --- Streamed Response ---
async def stream_response(input_data, history, is_audio=False, source_lang="twi"):
    history = history or []
    lang_map = {"twi": "tw", "ga": "ga", "ewe": "ee", "hausa": "ha", "english": "en"}
    source_code = lang_map.get(source_lang.lower(), "tw")
    lang_pair = f"{source_code}-en"

    if is_audio and isinstance(input_data, str):
        raw_text = transcribe_audio(input_data)
        print(f"[Transcribed] {raw_text}")
        user_message = translate_text(raw_text, target_lang=lang_pair)
        print(f"[Translated] {user_message}")
    else:
        user_message = input_data

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": ""})

    docs = await retriever.ainvoke(user_message)
    knowledge = "".join([doc.page_content + "\n\n" for doc in docs])

    # Format chat history
    chat_history_text = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in history if msg["role"] in {"user", "assistant"}
    )

    rag_prompt = f"""
You are an assistant answering based on the knowledge below.
If the knowledge is not enough, rely on your own logic.
Avoid saying you used knowledge or were trained on it.
Be concise and avoid repeating the question.

Question: {user_message}

Conversation history: {chat_history_text}

Knowledge:
{knowledge}
"""

    accumulated = ""
    async for chunk in llm.astream(rag_prompt):
        accumulated += chunk.content
        history[-1]["content"] = accumulated
        yield history, history

