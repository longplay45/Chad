# app.py

# Import necessary libraries and modules
import time
from pathlib import Path

import magic
import streamlit as st

import chad_with_docs as docs_chain
import chad_with_images as img_chain
import chad_with_memory as mem_chain
from chad_conversation_types import load_conversation_type
from ollama_helpers import (
    get_local_models,
    get_total_ram,
    is_mps_supported,
    is_resource_available,
)
from vectorstore import count_all_documents_in_vectorstore

# Constants
OLLAMA_API = "http://localhost:11434"
USER_AVATAR = '🙋'  # User avatar icon
AI_AVATAR = '👨'   # AI avatar icon
AI_CURSOR = '▌'    # Cursor icon used in chat
DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_SYSTEM_PROMPT = load_conversation_type()


def init_page():
    """Configure Streamlit page settings (title, layout, etc.)."""
    st.set_page_config(
        page_title="Hi, I'm Chad...", layout="wide", initial_sidebar_state="auto",
    )
    # Hide deploy button
    st.markdown(
        r"""
        <style>
        .stDeployButton {
                visibility: hidden;
            }
        </style>
        """, unsafe_allow_html=True
    )

def init_session():
    """Initialize session state variables for conversation and uploads."""
    if "conversation_type" not in st.session_state:
        st.session_state.conversation_type = "conversation"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "upload_counter" not in st.session_state:
        st.session_state.upload_counter = 0
    if "upload_full_path" not in st.session_state:
        st.session_state.upload_full_path = None
    if "upload_mime_type" not in st.session_state:
        st.session_state.upload_mime_type = None

    num_gpu = 1 if is_mps_supported() else 0
    if "conversation" not in st.session_state:
        st.session_state.conversation = init_default_chain(num_gpu)


def set_conversation_type(conversation_type):
    """Set the conversation type."""
    st.session_state.conversation_type = conversation_type


def add_message(role, message):
    """Add a single message to chat history."""
    st.session_state.messages.append({"role": role, "content": message})


def reset_conversation():
    """Reset the chat history to start a new conversation."""
    st.session_state.messages = []


def handle_upload(uploaded_file):
    """Handle file uploads and determine the file type."""
    save_folder = "tmp"
    save_path = Path(save_folder, uploaded_file.name)
    with open(save_path, mode="wb") as f:
        f.write(uploaded_file.getvalue())
        st.session_state.upload_full_path = f.name

    st.session_state.upload_counter += 1

    mime_type = magic.Magic()
    file_content = uploaded_file.read()
    content_type = mime_type.from_buffer(file_content)

    if "image data" in content_type.lower():
        st.session_state.upload_mime_type = "image"

    if "pdf" in content_type.lower():
        st.session_state.upload_mime_type = "pdf"

    if st.session_state.upload_mime_type == "image":
        st.image(uploaded_file)


def render_header():
    """Display the header of the application in the Streamlit interface."""
    st.markdown(f"""**Hi, my name is > Chad {AI_AVATAR}.**""")
    st.markdown(
        "_You can ask me anything about the world, programming, your documents or even your images......_"
    )


def render_settings():
    """Display and handle settings related to the AI model and conversation."""
    local_models = get_local_models()
    default_model = DEFAULT_MODEL
    default_ix = local_models.index(default_model)
    # with st.expander("Advanced settings"):
    model_name = st.selectbox("Select LLM", local_models, index=default_ix)
    gpu_on = st.toggle("Activate GPU", is_mps_supported())
    if gpu_on:
        gpu_on = 1
    if st.button("Reset conversation"):
        reset_conversation()

    return model_name, gpu_on


def render_upload():
    """Create an upload interface for files and handle the upload logic."""
    uploaded_file = st.file_uploader("Upload Files", type=["pdf", "jpg", "png"])

    if uploaded_file:
        handle_upload(uploaded_file)

        ai_image_message = f"I 👁️, you've just uploaded the file _**{uploaded_file.name}**_. Do you have any question about it?"
        if st.session_state.upload_counter == 1:
            add_message("ai", ai_image_message)
    else:
        st.session_state.upload_counter = 0
        st.session_state.upload_full_path = None
        st.session_state.upload_mime_type = None


def render_system_info():
    """Display system information such as RAM and GPU availability."""
    mps_support_info = (
        "MPS (GPU) available" if is_mps_supported() else "MPS (GPU) not available"
    )
    st.code(
        f"""
        Total RAM {get_total_ram()} GB
        {count_all_documents_in_vectorstore()} Docs in Vectorstore
        {mps_support_info}"""
    )


def render_conversation():
    """Render the conversation history in the chat interface."""
    for message in st.session_state.messages:
        role = message["role"]
        avatar = AI_AVATAR if role == "ai" else USER_AVATAR
        with st.chat_message(role, avatar=avatar):
            st.markdown(message["content"], unsafe_allow_html=True)

def render_chat(conversation):
    """Display the chat interface and handle chat interactions."""
    render_conversation()

    conversation = st.session_state.conversation
    # Accept user input
    if question := st.chat_input("What's on your mind? Ask me anything!"):
        # Add user message to chat history
        add_message("user", question)

        # Display user message in chat message container
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(question)

        # Display ai response in chat message container
        with st.chat_message("ai", avatar=AI_AVATAR):
            message_placeholder = st.empty()

            full_response = ""
            with st.spinner("Thinking..."):
                if st.session_state.upload_counter > 0:
                    if st.session_state.upload_mime_type == "image":
                        full_response = conversation.invoke(question)
                    else:
                        full_response = conversation({"query": question})["result"]
                else:
                    full_response = conversation.run(question=question)
                    full_response = full_response.replace("ChatBot", "**> Chad 👨**")

            # Simulate stream of response with milliseconds delay
            full_response_part = ""
            for chunk in full_response.split():
                full_response_part += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response_part + AI_CURSOR)

            message_placeholder.markdown(full_response)

        # Add ai response to chat history
        add_message("ai", full_response)


def render_api_unavailable_message():
    """Display a message when the OLLAMA API is not available."""
    st.markdown(
        f"""
        ## Ups.
        The API seems not to be available under {OLLAMA_API}. 
        
        Have you started the 🦙 **Ollama** app or via the command line?

        ```
        ollama serve
        ```
    """
    )


def init_default_chain(num_gpu):
    return mem_chain.init_chain(
        DEFAULT_MODEL,
        DEFAULT_TEMPERATURE,
        DEFAULT_SYSTEM_PROMPT,
        num_gpu=num_gpu,
        verbose=False,
    )


def select_chain(model_name, gpu_on):
    if st.session_state.upload_counter > 0:
        if st.session_state.upload_mime_type == "image":
            # Initialize the image chain
            model = img_chain.init_model("bakllava", num_gpu=gpu_on)
            conversation = img_chain.init_chain(
                file_path=st.session_state.upload_full_path, model=model
            )
            st.session_state.conversation = conversation
        else:
            # Initialize the document chain
            prompt = docs_chain.init_prompt()
            model = docs_chain.init_model(model_name, num_gpu=gpu_on)
            conversation = docs_chain.init_chain(prompt, model)
            st.session_state.conversation = conversation
    else:
        # Initialize the conversation chain
        if (
            not st.session_state.conversation
            and st.session_state.upload_mime_type == None
        ):
            conversation = mem_chain.init_chain(
                model_name,
                DEFAULT_TEMPERATURE,
                None, # DONT OVERWRITE MODEL SYSTEM PROMP
                num_gpu=gpu_on,
                verbose=False,
            )
            st.session_state.conversation = conversation


def main():
    """Main function to run the Streamlit application."""
    init_page()
    if not is_resource_available(OLLAMA_API):
        render_api_unavailable_message()
    else:
        init_session()

        # Render sidebar
        with st.sidebar:
            render_header()
            with st.expander("PDFs & Images"):
                render_upload()
            with st.expander("Advanced Settings"):
                model_name, gpu_on = render_settings()
            with st.expander("System Info"):
                render_system_info()

        # Select conversation chain
        conversation = select_chain(model_name, gpu_on)

        # Display chat messages from history on app rerun
        render_chat(conversation)


if __name__ == "__main__":
    main()
