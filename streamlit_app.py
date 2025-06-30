import os
import nest_asyncio
from dotenv import load_dotenv

import streamlit as st
from streamlit_chat import message

from llama_index.core.memory import Memory
from agent import agent  


nest_asyncio.apply()
load_dotenv()


st.set_page_config(
    page_title="IT Support Agent",
    page_icon="💻"
)


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hello! How can I help you with TechCorp IT today?"}
    ]

if "memory" not in st.session_state:
   
    st.session_state.memory = Memory.from_defaults(token_limit=30000)


agent.memory = st.session_state.memory


for msg in st.session_state.messages:
    message(msg["content"], is_user=(msg["role"] == "user"))


if user_query := st.chat_input("How can I help you with IT support today?"):
   
    st.session_state.messages.append({"role": "user", "content": user_query})
    message(user_query, is_user=True)

    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = agent.chat(user_query)
                reply = resp.response if hasattr(resp, "response") else str(resp)
            except Exception as e:
                reply = f"⚠️ Sorry, I encountered an error: {e}"
        message(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})


with st.sidebar:
    st.header("About")
    st.markdown("""
    This IT Support Agent can help you with:

    - **Company IT Policies**  
    - **Installation Guides**  
    - **IT Support Categories**  
    - **Knowledge Base**  
    - **Troubleshooting Flows**  
    """)
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.memory = Memory.from_defaults(token_limit=30000)
        agent.memory = st.session_state.memory
        st.experimental_rerun()


st.markdown("---")
st.markdown("*Powered by LlamaIndex & NVIDIA NIM*")
