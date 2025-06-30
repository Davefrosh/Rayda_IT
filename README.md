# TechCorp IT-Support Agent

An end-to-end LlamaIndex‐powered chatbot for TechCorp’s IT support knowledge base.  
Users can ask questions about company IT policies, installation guides, support categories, knowledge-base articles, sample conversations, test-request examples, and troubleshooting flows via a friendly chat UI (Streamlit) or a REST API (FastAPI).



## 📦 Project Structure

.
├── agent.py # Builds LlamaIndex indexes, tools & ReAct agent
├── streamlit_app.py # Streamlit front-end with memory and chat UI
├── fast_api.py # FastAPI wrapper exposing /chat endpoint
├── docs/
│ ├── company_it_policies.pdf
│ ├── installation_guides.pdf
│ ├── it_support_categories.pdf
│ ├── knowledge_base.pdf
│ ├── sample_conversations.pdf
│ ├── test_requests.pdf
│ └── troubleshooting_database.pdf
├── requirements.txt
└── README.md


⚙️ 1. Document Ingestion & Indexing
🛠️ 2. Wrapping as Tools
🤖 3. ReAct Agent Setup
🖥️ 4. Streamlit Chat App
🌐 5. FastAPI REST Endpoint
