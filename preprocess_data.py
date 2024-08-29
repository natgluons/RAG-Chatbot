import openai
import numpy as np
import faiss
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# CV portfolio data
portfolio_data = [
    {
        "id": 1,
        "text": "Data Analyst at LinkAja",
        "details": "Achieved 4/4 project completion across model development cycles, including implementing PCA and Isolation Forest for anomaly detection, applying Random Forest and Network Analysis to map illegal online gambling transactions, developing semi-supervised Relational Graph Convolution Network (RGCN) model for collusion risk assessment, and creating a gradient boosting model utilizing Heterogeneous Graph Transformer (HGT) architecture to detect syndicate fraudsters within clustered graphs, contributing to best practices in Anti-Money Laundering (AML). Utilized GCP cloud automation tools for model deployment and contributed to reducing fraud incidents by up to 11.2%, saving 95% in potential losses each month."
    },
    {
        "id": 2,
        "text": "Data Scientist at Kalbe Nutritionals",
        "details": "Utilized PostgreSQL for daily data exploration and developed interactive Tableau dashboards for monitoring. Achieved 100% completion in 2 sprints by employing ARIMA time-series regression for daily product quantity estimation and applying K-Means Clustering to optimize marketing strategies and provide personalized promotions based on customer segments."
    },
    {
        "id": 3,
        "text": "Artificial Intelligence Engineer / NLP Developer at Tikus Beken",
        "details": "Designed and deployed a Flask-based recommendation system API incorporating NLP techniques such as topic modeling, named entity recognition, and sentiment analysis, improving personalized job matching between seekers and listers by 30%. Utilized Docker and Kubernetes for deployment, enhancing system scalability."
    },
    {
        "id": 4,
        "text": "Machine Learning Researcher at FITB Research Grant ITB",
        "details": "Conducted research focused on time-series and Artificial Neural Network (ANN) applications."
    },
    {
        "id": 5,
        "text": "Full-Stack Developer Internship at BTPN Syariah",
        "details": "Worked on various full-stack development projects, implementing secure user registration and profile management APIs using Gin Gonic and JWT authentication."
    },
    {
        "id": 6,
        "text": "Front-End Developer Internship at Core Initiative Studio",
        "details": "Developed front-end components for web applications using modern JavaScript frameworks."
    },
    {
        "id": 7,
        "text": "Data Science Research Internship at LAPAN/Indonesian National Aeronautics & Space Administration",
        "details": "Worked on data science research projects related to aeronautics and space."
    },
    {
        "id": 8,
        "text": "GIS Data Analyst Internship at Garda Caah",
        "details": "Conducted GIS data analysis for various projects."
    },
    {
        "id": 9,
        "text": "Fraud Detection System API: Syndicate Indication using Network Graph Analysis - Hackathon BI 2024",
        "details": "Developed a Flask API to generate JSON files of node embeddings (graph positions), syndicate scores, and clusters."
    },
    {
        "id": 10,
        "text": "YouTube NLP Video Recommendation Algorithm using User’s Watch History",
        "details": "Utilized YouTube API with OAuth 2.0 to extract keywords using NLTK within a Kedro data pipeline and provide recommendations."
    },
    {
        "id": 11,
        "text": "Mobile Banking API Development: Secure User Registration and Profile Management",
        "details": "Developed secure user registration API for a mobile banking application using Gin Gonic, implementing JWT authentication."
    },
    {
        "id": 12,
        "text": "Automated Customer Churn Prediction System (MySkill x Deloitte)",
        "details": "Developed a decision tree model prediction and automated it via cron job scheduling on VM, visualizing results in Power BI."
    }
]

# Create embeddings
def get_embeddings(texts):
    response = openai.Embedding.create(
        input=texts,
        model="text-embedding-ada-002"
    )
    embeddings = [embedding['embedding'] for embedding in response['data']]
    return np.array(embeddings)

texts = [item["text"] for item in portfolio_data]
embeddings = get_embeddings(texts)

# Save embeddings and portfolio data
np.save("embeddings.npy", embeddings)
np.save("portfolio_data.npy", portfolio_data)
