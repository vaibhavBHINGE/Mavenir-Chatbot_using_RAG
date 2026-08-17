import streamlit as st
import requests
import uuid
import json

# --- 1. Page Configuration (Must be first) ---
st.set_page_config(
    page_title="3GPP Knowledge Agent | Mavenir",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Custom CSS for Enterprise Look ---
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #f0f2f6;
        border-left: 5px solid #2e7bcf;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. Backend Configuration ---
# This must match the port where Uvicorn is running
API_URL = "http://127.0.0.1:8000/chat"
HEALTH_URL = "http://127.0.0.1:8000/"

# --- 4. Session State Management ---
if "session_id" not in st.session_state:
    # Generate a unique ID for this user's browser session
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    # Initialize with a greeting from the bot
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to the 3GPP Technical Support Agent. I can search the 1510-page telecommunications specification and escalate complex issues. How can I assist you today?"}
    ]

# --- 5. Sidebar Layout (Controls & Status) ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Mavenir_Logo.svg/512px-Mavenir_Logo.svg.png", width=150)
    st.markdown("### 📡 3GPP Knowledge Agent")
    st.markdown("Retrieval-Augmented Generation (RAG) Architecture built for zero-hallucination document querying.")
    
    st.divider()
    
    # System Status Indicator
   # System Status Indicator
    st.markdown("#### System Status")
    try:
        # Increase timeout significantly and handle the connection more gracefully
        res = requests.get(HEALTH_URL, timeout=15) 
        if res.status_code == 200:
            st.success("Backend API: Online")
        else:
            st.warning(f"Backend API: Unresponsive (Code: {res.status_code})")
    except requests.exceptions.Timeout:
        st.error("Backend API: Timed out (Server is too slow)")
    except requests.exceptions.ConnectionError:
        st.error("Backend API: Offline (Run: uvicorn Backend.main:app --reload)")
    
    # Session Controls
    st.markdown("#### Session Controls")
    st.caption(f"Session ID: `{st.session_state.session_id[:8]}...`")
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        # We also reset the session ID so the backend starts a fresh JSON file
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# --- 6. Main Chat Interface ---
st.title("3GPP Technical Support")
st.markdown("""
<div style="padding: 12px 16px; border-radius: 8px; background-color: #1e293b; border: 1px solid #334155; color: #f8fafc; margin-bottom: 20px;">
    <b>Active Knowledge Base:</b> 23501-j80.docx (1,510 Pages) &nbsp;|&nbsp; <b>Mode:</b> Strict Retrieval (Zero Hallucination)
</div>
""", unsafe_allow_html=True)
# Render the conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 7. Chat Input & API Execution ---
if prompt := st.chat_input("Ask a question about the 3GPP specifications..."):
    # 1. Immediately display the user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Call the FastAPI Backend
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            with st.spinner("Searching 3GPP Standards..."):
                payload = {
                    "session_id": st.session_state.session_id,
                    "message": prompt
                }
                
                # Send the POST request to your FastAPI server
                response = requests.post(API_URL, json=payload)
                response.raise_for_status() # Check for HTTP errors
                
                # Extract the reply string from the JSON response
                bot_reply = response.json().get("reply", "Error: No reply found in response.")
                
                # Display the reply
                message_placeholder.markdown(bot_reply)
                
                # Save to Streamlit memory so it stays on screen after a rerun
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                
        except requests.exceptions.ConnectionError:
            error_msg = "🚨 Connection Error: Cannot reach the backend API. Is Uvicorn running on port 8000?"
            message_placeholder.error(error_msg)
        except Exception as e:
            error_msg = f"🚨 An error occurred: {str(e)}"
            message_placeholder.error(error_msg)