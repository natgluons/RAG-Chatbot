
# OpenAI Chatbot with RAG (Retrieval-Augmented Generation): Retrieve Information from CVOpenAI Chatbot with RAG (Retrieval-Augmented Generation): Retrieve Information from CV

## Overview
This project is an OpenAI-powered chatbot that uses Retrieval-Augmented Generation (RAG) to enhance responses with relevant information from provided documents. The chatbot is deployed using Docker and Kubernetes, and it's designed to be accessed via a web interface.

## Features
- Natural Language Processing: Utilizes OpenAI's GPT-3.5 model to generate responses.
- Document Retrieval: Enhances responses by retrieving relevant information from a set of documents stored in the `knowledge_sources` folder.
- Web Interface: Provides a simple web interface for user interaction.
- Dockerized Deployment: Containerized using Docker for easy deployment.
- Kubernetes: Supports deployment on Kubernetes for scalable and reliable service.

## Requirements
- Python 3.9+
- Flask==2.0.3
- Werkzeug==2.0.3
- openai==1.38.0
- sqlalchemy==1.4.25
- python-dotenv==1.0.1
- PyPDF2==3.0.1
- pandas==2.2.0
- scikit-learn==1.5.0
- Docker
- Kubernetes

## Access the Web Interface
Open your browser and navigate to `http://34.71.245.123/` to interact with the chatbot. Ask anything related to my CV, background, professional, and academic experience. This is the minimum viable product (MVP) under development; the final version will be hosted on a domain website, to be announced later.

*this service is currently offline due to cost considerations (Why does Kubernetes cost so much!?)

![ragchatbot](https://github.com/user-attachments/assets/4570bf02-735f-4f92-94f8-b803e6859997)
