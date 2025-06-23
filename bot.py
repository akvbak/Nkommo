import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI
from ghana_nlp import GhanaNLP
from rasa_client import parse_intent_with_rasa
import api
import voice
import data

# --- Configuration ---
client = OpenAI(api_key=api.APIKEY)
os.environ["OPENAI_API_KEY"] = api.APIKEY
os.environ["GHANA_NLP_API_KEY"] = voice.APIKEY

# Paths
datapath = r"files"
chromapath = r"chroma_db"

# Embeddings & Model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(model='gpt-4o', temperature=0.5)
nlp = GhanaNLP(voice.APIKEY)

# Chroma vectorstore
vector_store = Chroma(
    collection_name="EZPROMO",
    embedding_function=embeddings_model,
    persist_directory=chromapath,
)
retriever = vector_store.as_retriever(search_kwargs={'k': 5})

# --- GhanaNLP Helpers ---
def transcribe_audio(filepath: str, source_lang: str) -> str:
    if source_lang.lower() == "english":
        with open(filepath, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        return transcript.text
    else:
        response = nlp.stt(filepath)
        print(f"[DEBUG] GhanaNLP STT Response: {response}")

        if isinstance(response, str):
            return response

        elif isinstance(response, dict):
        # Handle known key formats
            if "text" in response:
                return response["text"]
            elif "translation" in response:
                return response["translation"]
            elif "message" in response:
                print(f"[WARN] GhanaNLP returned message: {response['message']}")
            return response["message"]

        # fallback to string version of response
        return str(response)


#def translate_text(text, target_lang="tw-en"):
#    response = nlp.translate(text, target_lang)
#    if isinstance(response, str):
        return response
#    elif isinstance(response, dict):
#        if "translation" in response:
#            return response["translation"]
#        if "message" in response:
#            print(f"[WARN] Translation failed: {response['message']}")
#    return text

# --- Core Stream Response ---
def stream_response(input_data, history, is_audio=False, source_lang="twi"):
    history = history or []
    lang_map = {"english":"en",
                "twi":"tw",
                "ga":"ga",
                "ewe":"ee",
                "hausa":"ha",
                "dagbani":"dag"}
    source_code = lang_map.get(source_lang.lower(), "tw")
    lang_pair = f"{source_code}-en"

    if is_audio and isinstance(input_data, str):
        raw_text = transcribe_audio(input_data, source_lang=source_lang)
        print(f"[Transcribed] {raw_text}")
#        user_message = translate_text(raw_text, target_lang=lang_pair)
#        print(f"[Translated] {user_message}")
        user_message = raw_text
    else:
        user_message = input_data

    history.append((user_message, ""))

    # Rasa intent parsing
    sender_id = "Promogo"
    intent, text_resp, conf = parse_intent_with_rasa(sender_id, user_message, source_code)
    print(f"[Rasa] intent={intent}, confidence={conf}")
    if intent and conf > 0.7:
        history[-1] = (user_message, text_resp or "")
        yield history, history
        return

    # Fallback to RAG
    docs = retriever.invoke(user_message)
    knowledge = "".join([d.page_content + "\n\n" for d in docs])
    rag_prompt = f"""
You are an assistant which answers questions based on knowledge provided.
You can also answer questions based on the conversation history.
You can also ask clarifying questions if the user is not clear.
You should treat every instance of the words 'Pramogana', PromoGhana and EZPromo as 'PromoGo'.
The question: {user_message}
Conversation history: {history}
The knowledge: {knowledge}
"""
    accumulated = ""
    for chunk in llm.stream(rag_prompt):
        accumulated += chunk.content
        history[-1] = (user_message, accumulated)
        yield history, history