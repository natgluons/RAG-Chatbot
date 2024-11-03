from flask import Flask, request, jsonify, send_from_directory
from logging.config import dictConfig
from models import get_db_connection, init_db
from openai import OpenAI
import os
import pandas as pd
from dotenv import load_dotenv
import time

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferMemory
from PyPDF2 import PdfReader

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})

app = Flask(__name__, static_folder='.', static_url_path='')
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'csv'}

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

init_db()

def chunk_text(text, max_length=50):
    words = text.split()
    return [' '.join(words[i:i + max_length]) for i in range(0, len(words), max_length)]

def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as file:
                text_chunks = chunk_text(file.read())
                for chunk in text_chunks:
                    documents.append({
                        "content": chunk,
                        "metadata": {
                            "source": filename,
                            "page": 1
                        }
                    })
        elif filename.endswith(".pdf"):
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    text_chunks = chunk_text(text)
                    for chunk in text_chunks:
                        documents.append({
                            "content": chunk,
                            "metadata": {
                                "source": filename,
                                "page": page_num + 1
                            }
                        })
        elif filename.endswith(".csv"):
            df = pd.read_csv(file_path)
            text_chunks = chunk_text(df.to_string())
            for chunk in text_chunks:
                documents.append({
                    "content": chunk,
                    "metadata": {
                        "source": filename,
                        "page": 1
                    }
                })
    return documents

folder_path = 'knowledge_sources'
documents = load_documents(folder_path)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
app.logger.info(f"{len(documents)} documents loaded")
app.logger.info("FAISS indexing...")
start_time = time.time()
faiss_index = FAISS.from_texts(
    [doc['content'] for doc in documents], 
    embedding=embeddings,
    metadatas=[doc['metadata'] for doc in documents]
)
app.logger.info(f"FAISS indexing done in {time.time() - start_time} seconds")

retriever = faiss_index.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.7
    }
)

# Initialize conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="result"
)

qa_chain = RetrievalQA.from_llm(
    llm=ChatOpenAI(
        model="gpt-4", 
        api_key=openai_api_key,
        max_tokens=500,
        temperature=0.7
    ),
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    verbose=True,
    output_key="result"
)

def save_prompt(user_id, role, prompt):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO user_prompts (user_id, role, prompt) VALUES (?, ?, ?)", (user_id, role, prompt))
    conn.commit()
    conn.close()

def get_last_prompts(user_id, limit=10):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT role, prompt FROM user_prompts WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?", (user_id, limit))
    prompts = cur.fetchall()
    conn.close()
    return [{"role": p[0], "content": p[1]} for p in prompts][::-1]

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    prompt = data['prompt']
    user_id = 1

    save_prompt(user_id, "user", prompt)

    start_retrieval_time = time.time()
    response = qa_chain.invoke({"query": prompt})
    retrieval_time = time.time() - start_retrieval_time

    assistant_message = response["result"]
    save_prompt(user_id, "assistant", assistant_message)

    start_generation_time = time.time()
    generation_time = time.time() - start_generation_time

    # Extract source documents from response
    retrieved_docs = response.get("source_documents", [])
    structured_response = {
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "source_documents": [
            {
                "title": doc.metadata.get('source', 'Unknown'),
                "page_number": doc.metadata.get('page', 'N/A')
            }
            for doc in retrieved_docs
        ]
    }

    # Filter out duplicate sources
    seen_sources = set()
    unique_sources = []
    for source in structured_response['source_documents']:
        source_key = f"{source['title']}-{source['page_number']}"
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            unique_sources.append(source)
    
    structured_response['source_documents'] = unique_sources

    return jsonify({
        "response": assistant_message,
        "sources": structured_response['source_documents']
    })

@app.route('/')
def index():
    return send_from_directory('..', 'index.html')

@app.route('/app.js')
def app_js():
    return send_from_directory('..', 'app.js')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
