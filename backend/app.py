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
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
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
    output_key="result"  # Changed to be consistent
)

# Define an intent analysis chain to check if retrieval is needed
intent_analysis_prompt = PromptTemplate.from_template("""
Based on the following conversation and user query, please answer:
1. Should additional information from a knowledge base be retrieved? (Yes or No)
2. If yes, what is the main topic, entity, or name that the query refers to?

Conversation:
{chat_history}

User Query:
{user_query}

Answer:
""")

intent_analysis_chain = LLMChain(
    llm=ChatOpenAI(model="gpt-4", api_key=openai_api_key, max_tokens=50),
    prompt=intent_analysis_prompt
)

# Define the RetrievalQA chain for retrieving and answering based on external documents
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(
        model="gpt-4", 
        api_key=openai_api_key,
        max_tokens=500,
        temperature=0.7
    ),
    chain_type="stuff",
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    chain_type_kwargs={
        "verbose": True,
        "prompt": PromptTemplate(
            template="""Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know.

            {context}

            Question: {question}
            Answer: """,
            input_variables=["context", "question"]
        )
    }
)

def save_prompt(user_id, role, prompt):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO user_prompts (user_id, role, prompt) VALUES (?, ?, ?)", (user_id, role, prompt))
    conn.commit()
    conn.close()

def parse_analysis_response(response_text):
    # Split the response by lines and clean each line
    lines = [line.strip() for line in response_text.splitlines() if line.strip()]
    
    # Initialize variables
    should_retrieve = False
    retrieval_topic = None

    # Loop through the lines and identify key parts
    for line in lines:
        if line.startswith("1."):
            if "Yes" in line:
                should_retrieve = True
            elif "No" in line:
                should_retrieve = False
        elif line.startswith("2."):
            retrieval_topic = line[3:].strip()  # Extract topic after "2."

    return should_retrieve, retrieval_topic


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    prompt = data['prompt']
    user_id = 1

    save_prompt(user_id, "user", prompt)

    # Load chat history
    chat_history = memory.load_memory_variables({})["chat_history"]
    chat_history_text = " ".join([turn.content for turn in chat_history])

    # Run intent analysis to check if retrieval is needed
    analysis_response = intent_analysis_chain.run({
        "chat_history": chat_history_text,
        "user_query": prompt
    })

    app.logger.info(f"Analysis Response: {analysis_response}")

    should_retrieve, retrieval_topic = parse_analysis_response(analysis_response)

    try:
        if should_retrieve:
            # Execute retrieval and QA based on user query
            response = qa_chain({"query": prompt})
            assistant_message = response["result"]
            sources = response.get("source_documents", [])
        else:
            # For non-retrieval queries, use a simple response from the LLM
            llm = ChatOpenAI(
                model="gpt-4", 
                api_key=openai_api_key,
                max_tokens=500,
                temperature=0.7
            )
            assistant_message = llm.predict(prompt)
            sources = []

        save_prompt(user_id, "assistant", assistant_message)

        # Format response with sources if retrieval was performed
        structured_response = {
            "source_documents": [
                {
                    "title": doc.metadata.get('source', 'Unknown'),
                    "page_number": doc.metadata.get('page', 'N/A')
                }
                for doc in sources
            ]
        }

        return jsonify({
            "response": assistant_message,
            "sources": structured_response['source_documents']
        })

    except Exception as e:
        app.logger.error(f"Error occurred: {e}")
        return jsonify({"error": "An error occurred while processing your request."}), 500
    
@app.route('/')
def index():
    return send_from_directory('..', 'index.html')

@app.route('/app.js')
def app_js():
    return send_from_directory('..', 'app.js')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
