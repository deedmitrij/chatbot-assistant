# 🗎 AI-Powered Hotel Support Assistant

This is a **RAG chatbot** designed for automated hotel guest support. It utilizes an **Orchestrator Pattern** to coordinate between **Vector DB search** and **LLM**, featuring a **Human-in-the-Loop** mechanism to handle low-confidence queries.

<img src="https://github.com/user-attachments/assets/b0c431dd-c237-41b9-a1fd-80936c6012e9" alt="Chatbot assistant screenshot" width="300">

---

## 📑 Table of Contents
1. [Architecture Overview](#-architecture-overview)
   - [Retrieval Layer](#1️⃣-retrieval-layer)
   - [Orchestration Layer](#2️⃣-orchestration-layer)
   - [LLM Layer](#3️⃣-llm-layer)
   - [Human-in-the-Loop (HITL)](#4️⃣-human-in-the-loop-hitl)
2. [Quality Assurance & Testing](#%EF%B8%8F-quality-assurance--testing)
3. [RAGAS Evaluation](#-ragas-evaluation)
4. [Tech Stack](#-tech-stack)
5. [Prerequisites](#-prerequisites)
6. [Setup Instructions](#%EF%B8%8F-setup-instructions)
   - [Clone the Repository](#1️⃣-clone-the-repository)
   - [Set Up a Virtual Environment](#2️⃣-set-up-a-virtual-environment)
   - [Install Dependencies](#3️⃣-install-dependencies)
   - [Configure Environment Variables](#4️⃣-configure-environment-variables)
   - [Run the Application](#5️⃣-run-the-application)
   - [Running Tests](#6️⃣-running-tests)
7. [API Setup](#-api-setup)
   - [Telegram Bot](#-telegram-bot)
   - [Hugging Face Inference API](#-hugging-face-inference-api)
8. [How It Works](#-how-it-works)
9. [License](#%EF%B8%8F-license)

---

## 🧱 Architecture Overview
This application is built as an **AI-powered RAG (Retrieval-Augmented Generation) system** using a centralized Orchestrator to manage data flow and logic.

### 1️⃣ Retrieval Layer
1. **Knowledge Base**: Uses a structured `knowledge_base.json` as the primary source of resort information.
2. **Vector Storage**: Text chunks are embedded and stored in a **ChromaDB** index for high-speed semantic similarity search.

### 2️⃣ Orchestration Layer
1. **ChatManager**: Acts as the "Brain" of the operation. It manages the lifecycle of a message:
   - Triggers embedding of the user query.
   - Queries the Vector DB for context.
   - Evaluates the **Confidence Score**.
   - Decides whether to answer directly or route to a human operator.

### 3️⃣ LLM Layer
1. **Text Generation**: Powered by **Qwen 2.5 (7B Instruct)** via the Hugging Face Router.
2. **OpenAI SDK**: Used as a robust interface to interact with remote inference endpoints.
3. **Role-Play**: Strict system prompt ensure the AI maintains a "Hotel Concierge" persona using corresponding identity.

### 4️⃣ Human-in-the-Loop (HITL)
1. **Threshold Logic**: If the vector search returns a confidence score below the threshold, the system triggers a "pending approval" state.
2. **Operator Alerts**: Designed to integrate with Telegram to allow hotel staff to review AI suggestions and intervene in real-time.

---

## 🕵 Quality Assurance & Testing
The project includes a comprehensive **Automated Testing Framework** to prevent hallucinations and maintain "Brand Voice":

1. **LLM-as-a-Judge**: evaluates the assistant's performance across multiple categories.
- **Groundedness (Faithfulness)**: Ensuring answers are strictly based on the provided context.
- **Negative Constraints**: Verifying the AI admits ignorance when information is missing instead of hallucinating.
- **Relevancy & Completeness**: Checking if all parts of a user query are addressed.
- **Tone & Persona**: Monitoring "Brand Voice" consistency (politeness).

2. **Vector Database**: specialized tests to ensure the ChromaDB index and retrieval logic work with high precision:
- **Semantic Retrieval Accuracy**: Basic verification that the system retrieves the most relevant chunks for standard queries.
- **Top-K Recall Optimization**: Measuring if the "ground truth" information is consistently present within the top-K retrieved results.
- **Metadata Filtering**: Ensuring that search results can be correctly narrowed down using metadata tags without losing semantic relevance.

---

## 📊 RAGAS Evaluation
The project includes **RAGAS (Retrieval-Augmented Generation Assessment)** evaluation to measure the technical performance of our pipeline:

1. **Faithfulness**: Measures the factual consistency of the generated answer against the retrieved context.
2. **Answer Relevance**: Evaluates how well the answer addresses the user's specific query without redundant info.
3. **Context Precision**: Calculates the signal-to-noise ratio in the retrieved chunks (how relevant the top-K results are).
4. **Context Recall**: Checks if the retrieved context actually contains the ground-truth information needed to answer.

---

## 🧰 Tech Stack
- **Python** – Core logic
- **Flask** – Web framework and API routing
- **ChromaDB** – Vector database for similarity search
- **Hugging Face Hub** – Native inference client for embeddings
- **OpenAI Python SDK** – Client for LLM interactions
- **Telegram Bot API** – Operator interface
- **pytest** – Testing engine
- **RAGAS** – RAG evaluation framework

---

## 📋 Prerequisites
- Python 3.10+
- **Hugging Face account** with an **Access Token** (Write/Inference permissions)
- **Telegram Account**: To create a bot and receive operator alerts via the Telegram Bot API.
- **ngrok** (or similar): Required for local development to expose your webhook to Telegram's servers.
  
---

## 🛠️ Setup Instructions

### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/deedmitrij/chatbot-assistant.git
cd chatbot-assistant
```

### **2️⃣ Set Up a Virtual Environment**
```sh
python -m venv .venv
.venv\Scripts\activate
```

### **3️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

### **4️⃣ Configure Environment Variables**
Create a `.env` file in the project root and add the following:

```ini
# Telegram Configuration
TG_BOT_TOKEN=your_telegram_bot_token
TG_ADMIN_ID=your_telegram_chat_id

# Hugging Face Configuration
HF_API_TOKEN=your_huggingface_api_key
HF_BASE_URL=router_huggingface_url

# Model Selection
CHAT_MODEL=main_llm_model
EMBEDDING_MODEL=embedding_model
```
📌 **Note:**
- `TG_BOT_TOKEN`: Replace with the API token you received from @BotFather.
- `TG_ADMIN_ID`: Replace with your unique Telegram User ID (get it from @userinfobot).
- `HF_API_TOKEN`: Replace with your Hugging Face Access Token.
- `HF_BASE_URL`: Use the standard Hugging Face Inference API URL (https://router.huggingface.co/v1).
- `CHAT_MODEL`: Specify the model for text generation (e.g., Qwen/Qwen2.5-7B-Instruct).
- `EMBEDDING_MODEL`: Specify the model for embeddings (e.g., BAAI/bge-small-en-v1.5).


### **5️⃣ Run the Application**
```sh
python main.py
```

The chatbot will start and be accessible at **http://localhost:5000**.

### **6️⃣ Running Tests**
To run the automated QA suite:
```sh
pytest .\tests\
```

---

## 🔗 API Setup

### 🤖 Telegram Bot
To receive "Low Confidence" alerts and respond to guests from your phone:

1. **Create a Bot**: 
   - Message [@BotFather](https://t.me/botfather) on Telegram.
   - Use the `/newbot` command and follow the instructions.
   - Copy the **API Token** and add it to your `.env` file as `TG_BOT_TOKEN`.
2. **Get your Chat ID**: 
   - Message [@userinfobot](https://t.me/userinfobot).
   - Copy your unique **ID** (a string of numbers) and add it to your `.env` file as `TG_ADMIN_ID`.
3. **Initialize the Bot**: 
   - Open your new bot's chat and press **Start**. The bot cannot message you until you've interacted with it.
4. **Setup Webhook (Local Dev)**:
   - Use a tool like **ngrok** to create a public URL for your local server: `ngrok http 5000`.
   - Register the URL with Telegram:
     `https://api.telegram.org/bot<YOUR_TOKEN>/setWebhook?url=<YOUR_NGROK_URL>/webhook/telegram`

### 🤖 Hugging Face Inference API
To get access to Hugging Face models:

1. Visit **https://huggingface.co/**
2. Sign in or create a **Hugging Face account**.
3. Go to **Settings → Access Tokens**.
4. Create a new token with **Make calls to Inference Providers** permission.
5. Copy the token and add it to your `.env` file as `HF_API_KEY`.
   
---

## 🚀 How It Works  

1. **User Input**: A guest interacts with the chat by either selecting a predefined FAQ category or typing a natural language question into the interface.
2. **Answer Search**: The orchestrator transforms the query into a vector and performs a similarity search against the vector DB index to extract the most relevant policy or fact from the knowledge base.
3. **Response**: The system evaluates the proximity of the found data:
  - **Confidence High**: The system sends the question + context to LLM to generate a polite, branded response.
  - **Confidence Low**: The guest receives a bridging message, while the system alerts a human operator to review the AI-generated suggestion or provide a manual response.

---

## 🛡️ License
This project is **open-source** and available under the [MIT License](LICENSE).
