# 🗎 AI-Powered Hotel Support Assistant

This is a **RAG chatbot** designed for automated hotel guest support. It utilizes an **Orchestrator Pattern** to coordinate between vector search and LLM, featuring a **Human-in-the-Loop** mechanism to handle low-confidence queries.

<img src="https://github.com/user-attachments/assets/b0c431dd-c237-41b9-a1fd-80936c6012e9" alt="Chatbot assistant screenshot" width="300">

---

## 📑 Table of Contents
1. [Architecture Overview](#-architecture-overview)
   - [Retrieval Layer](#1️⃣-retrieval-layer)
   - [Orchestration Layer](#2️⃣-orchestration-layer)
   - [LLM & Embedding Layer](#3️⃣-llm--embedding-layer)
   - [Human-in-the-Loop (HITL)](#4️⃣-human-in-the-loop-hitl)
2. [Tech Stack](#-tech-stack)
3. [Prerequisites](#-prerequisites)
4. [Setup Instructions](#%EF%B8%8F-setup-instructions)
   - [Clone the Repository](#1️⃣-clone-the-repository)
   - [Set Up a Virtual Environment](#2️⃣-set-up-a-virtual-environment)
   - [Install Dependencies](#3️⃣-install-dependencies)
   - [Configure Environment Variables](#4️⃣-configure-environment-variables)
   - [Run the Application](#5️⃣-run-the-application)
5. [API Setup](#-api-setup)
   - [Hugging Face Inference API](#-hugging-face-inference-api)
6. [How It Works](#-how-it-works)
7. [License](#%EF%B8%8F-license)

---

## 🧱 Architecture Overview
This application is built as an **AI-powered RAG (Retrieval-Augmented Generation) system** using a centralized Orchestrator to manage data flow and logic.



### 1️⃣ Retrieval Layer
1. **Knowledge Base**: Uses a structured `knowledge_base.json` as the primary source of resort information.
2. **Vector Storage**: Text chunks are embedded and stored in a **FAISS** index for high-speed semantic similarity search.
3. **Retrieval Strategy**: Uses the **BGE-Small** model with specific task-prefixes (`query:` and `passage:`) to maximize search relevance.

### 2️⃣ Orchestration Layer
1. **ChatManager**: Acts as the "Brain" of the operation. It manages the lifecycle of a message:
   - Triggers embedding of the user query.
   - Queries the Vector DB for context.
   - Evaluates the **Confidence Score**.
   - Decides whether to answer directly or route to a human operator.

### 3️⃣ LLM & Embedding Layer
1. **Text Generation**: Powered by **Qwen 2.5 (7B Instruct)** via the Hugging Face Router.
2. **Embeddings**: Handled by **BAAI/bge-small-en-v1.5** via `InferenceClient`.
3. **OpenAI SDK**: Used as a robust interface to interact with remote inference endpoints.

### 4️⃣ Human-in-the-Loop (HITL)
1. **Threshold Logic**: If the vector search returns a confidence score below the threshold, the system triggers a "pending approval" state.
2. **Operator Alerts**: Designed to integrate with Telegram (Phase 4) to allow hotel staff to review AI suggestions and intervene in real-time.

---

## 🧰 Tech Stack
- **Python** – Core logic
- **Flask** – Web framework and API routing
- **FAISS** – Vector database for similarity search
- **Hugging Face Hub** – Native inference client for embeddings
- **OpenAI Python SDK** – Client for LLM interactions
- **NumPy** – Efficient vector and array manipulation

---

## 📋 Prerequisites
- Python 3.10+
- Hugging Face account with an **Access Token** (Write/Inference permissions)

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
HF_API_KEY=your_huggingface_api_key
```

📌 **Note:** Replace `your_huggingface_api_key` with your actual Hugging Face API key. 

### **5️⃣ Run the Application**
```sh
python main.py
```

The chatbot will start and be accessible at **http://localhost:5000**.

---

## 🔗 API Setup

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