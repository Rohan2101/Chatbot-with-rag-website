import asyncio
from numpy.lib.utils import source
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv

# Configure Streamlit theme to match LangChain docs
st.set_page_config(
    page_title="LangChain Doc Helper",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to match LangChain docs styling
st.markdown("""
    <style>
    /* Main background and text colors */
    .stApp {
        background-color: #1a1b1e;
        color: #ffffff;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #141517;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        background-color: #2a2b2e;
        color: #ffffff;
        border-color: #404040;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background-color: #2a2b2e;
    }
    
    /* Links */
    a {
        color: #00cf9d !important;
    }
    
    /* Divider color */
    .stDivider {
        border-color: #404040;
    }
    </style>
    """, unsafe_allow_html=True)

# Ensure the event loop is running
asyncio.set_event_loop(asyncio.new_event_loop())

from backend.core import run_llm
load_dotenv()

# Add sidebar with user info
with st.sidebar:
    st.title("User Profile")
    
    # Get user info if available
    if st.experimental_user.email:
        st.image("https://www.gravatar.com/avatar/00000000000000000000000000000000?d=mp&f=y", 
                width=100)
        st.write(f"👤 **Name:** {st.experimental_user.email.split('@')[0]}")
        st.write(f"📧 **Email:** {st.experimental_user.email}")
    else:
        st.write("Please login to view profile information")
    
    st.divider()

# Main content
st.header("LangChain Doc Helper")

prompt = st.text_input("Ask a question about LangChain", placeholder="Enter your prompt here")

if (
    "chat_answers_history" not in st.session_state and "user_prompt_history" not in st.session_state and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []

def create_sources_string(source_urls: set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i + 1}. {source}\n"
    return sources_string


if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(query=prompt,chat_history=st.session_state["chat_history"])
        sources = set(
            [doc.metadata["source"] for doc in generated_response["source_documents"]]
        )

        formatted_response = (
            f"{generated_response["result"]} \n\n {create_sources_string(sources)}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human",prompt))
        st.session_state["chat_history"].append(("ai",generated_response["result"]))


if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"]):
        message(user_query,is_user=True)
        message(generated_response)
