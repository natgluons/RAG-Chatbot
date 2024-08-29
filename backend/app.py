from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from models import get_db_connection, init_db           # do not change this!! use models, not backend.models
# import openai
from openai import OpenAI
import datetime
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
# from PyPDF2 import PdfFileReader
# from PyPDF2 import extract_text
import pandas as pd
from dotenv import load_dotenv

app = Flask(__name__, static_folder='.', static_url_path='')
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'csv'}

# Load environment variables from .env file
load_dotenv()
client = OpenAI()
# client = openai()

# Access the OpenAI API key
# OpenAI.api_key = os.getenv('OPENAI_API_KEY')
# openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize the database
init_db()

# Function to load documents from a folder
def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as file:
                documents.append(file.read())
        elif filename.endswith(".pdf"):
            reader = PdfReader(file_path)
            # reader = PdfFileReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            documents.append(text)
        elif filename.endswith(".csv"):
            df = pd.read_csv(file_path)
            documents.append(df.to_string())
    return documents

# # Function to load documents from a folder
# def load_documents(folder_path):
#     documents = []
#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)
#         if filename.endswith(".txt"):
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 documents.append(file.read())
#         elif filename.endswith(".pdf"):
#             with open(file_path, 'rb') as file:
#                 reader = PdfFileReader(file)
#                 text = ""
#                 for page_num in range(reader.getNumPages()):
#                     text += reader.getPage(page_num).extract_text()  # Change this to extract_text()
#                 documents.append(text)
#         elif filename.endswith(".csv"):
#             df = pd.read_csv(file_path)
#             documents.append(df.to_string())
#     return documents

# Load documents from the knowledge_sources folder
folder_path = 'knowledge_sources'  # replace with your folder path
documents = load_documents(folder_path)

# Vectorize documents using TF-IDF
vectorizer = TfidfVectorizer()
document_vectors = vectorizer.fit_transform(documents)

# Function to search documents
def search_documents(query, document_vectors, vectorizer):
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, document_vectors).flatten()
    related_docs_indices = cosine_similarities.argsort()[::-1]
    return related_docs_indices, cosine_similarities

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
    return [{"role": p['role'], "content": p['prompt']} for p in prompts][::-1]  # Reverse to get the oldest first

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    prompt = data['prompt']
    user_id = 1  # In a real application, you would use the authenticated user's ID

    # Save the current user prompt
    save_prompt(user_id, "user", prompt)

    # Retrieve the last 10 prompts
    last_prompts = get_last_prompts(user_id)

    # Search documents
    related_docs_indices, similarities = search_documents(prompt, document_vectors, vectorizer)
    document_results = [documents[idx] for idx in related_docs_indices[:3]]  # Top 3 documents
    document_texts = " ".join(document_results)

    # Prepare the conversation history
    messages = last_prompts + [{"role": "user", "content": prompt}]
    context_message = {"role": "system", "content": f"Relevant information: {document_texts}"}
    messages.insert(0, context_message)  # Insert context at the beginning

    response = client.chat.completions.create(
    # response = openai.ChatCompletion.create(
    # response = OpenAI.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    assistant_message = response.choices[0].message.content.strip()

    # Save the assistant's response
    save_prompt(user_id, "assistant", assistant_message)

    return jsonify(response=assistant_message)

@app.route('/')
def index():
    return send_from_directory('..', 'index.html')

@app.route('/app.js')
def app_js():
    return send_from_directory('..', 'app.js')

# @app.route('/<path:filename>')
# def serve_file(filename):
#     return send_from_directory('..', filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
