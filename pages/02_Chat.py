import streamlit as st
import base64
import os
import time
from dotenv import load_dotenv
from multimodal_rag import MultimodalRAG

# Load environment variables
load_dotenv()

# Function to check if database exists and has data
def check_database_exists():
    """Check if the database directory exists and has data"""
    db_path = "database"
    # Check for both vectorstore and document store
    vectorstore_exists = os.path.exists(db_path) and len(os.listdir(db_path)) > 0
    docstore_exists = os.path.exists(os.path.join(db_path, "docstore.sqlite"))
    return vectorstore_exists or docstore_exists

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_system" not in st.session_state:
    # Initialize RAG system if database exists
    if check_database_exists():
        st.session_state.rag_system = MultimodalRAG(persist_directory="database")
        st.session_state.files_processed = True
    else:
        st.session_state.rag_system = None
        st.session_state.files_processed = False
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []

# Function to clear chat history
def clear_chat_history():
    st.session_state.messages = []

# Main page header with custom styling
st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='color: #4257f5;'>💬 Welcome to DocuBot</h1>
        <p style='font-size: 1.2em; color: #666;'>Ask questions about your documents</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Sidebar for model selection and chat controls
with st.sidebar:
    st.markdown("### 🤖 Model Settings")
    model_name = st.selectbox(
        "Select OpenAI Model",
        ["gpt-4o", "gpt-4o-mini"],
        index=1  # Default to gpt-4o-mini as it's cheaper
    )
    
    st.markdown("---")
    
    # Add clear chat button
    st.button("🗑️ Clear Chat History", on_click=clear_chat_history, use_container_width=True)
    
    # Add link to document upload page
    st.markdown("### 📚 Document Management")
    st.info("📥 Go to the **Document Upload** to upload new documents.")

# Helper function to display base64 images
def display_base64_image(base64_code):
    image_data = base64.b64decode(base64_code)
    return image_data
    
# Check if documents have been processed
if not st.session_state.files_processed:
    if check_database_exists():
        # Initialize RAG system with existing database
        if st.session_state.rag_system is None:
            st.session_state.rag_system = MultimodalRAG(persist_directory="database")
            st.session_state.files_processed = True
        st.info("💡 Using existing document database. You can start chatting now.")
    else:
        st.warning("⚠️ Please upload and process documents first before chatting. Go to the **Document Upload** page in the sidebar.")

# Always display the chat interface regardless of document processing status
if st.session_state.files_processed:
    # Helper function to stream response text
    def stream_response(response_text):
        message_placeholder = st.empty()
        full_response = ""
        
        # Split by lines to preserve markdown formatting
        lines = response_text.split('\n')
        line_index = 0
        
        # Process line by line to preserve formatting
        for line in lines:
            # Process each line word by word
            words = line.split()
            for i, word in enumerate(words):
                full_response += word
                # Add space if not the last word of the line
                if i < len(words) - 1:
                    full_response += " "
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.01)  # Adjust speed as needed
            
            # Add newline if not the last line
            if line_index < len(lines) - 1:
                full_response += "\n"
            line_index += 1
            
        message_placeholder.markdown(full_response)
        return full_response
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
            # Display message content
            st.markdown(message["content"])
            
            # Display sources if available (for assistant messages only)
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                st.markdown("**Sources:**")
                source_text = ""
                for source in message["sources"]:
                    doc_name = source.get("document", "Unknown")
                    page_num = source.get("page", "Unknown")
                    source_text += f"- {doc_name} (Page: {page_num})\n"
                st.markdown(source_text)
            
            # Display images if available (for assistant messages only)
            if message["role"] == "assistant" and "images" in message and message["images"]:
                # Create columns for multiple images if needed
                if len(message["images"]) > 1:
                    cols = st.columns(min(len(message["images"]), 3))
                    for i, img_data in enumerate(message["images"]):
                        with cols[i % 3]:
                            st.image(img_data, use_container_width=True)
                else:
                    for img_data in message["images"]:
                        st.image(img_data)

# Get user input (placed at the bottom of the screen by Streamlit automatically)
user_question = st.chat_input("Ask a question about your documents...")

# Process user input and generate response
if user_question and st.session_state.files_processed:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_question})
    
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_question)
        # Generate response
    with st.chat_message("assistant", avatar="🤖"):
        # Get QA chain with sources
        chain_with_sources = st.session_state.rag_system.get_qa_chain_with_sources(model_name)
        
        # Get response
        with st.spinner("Thinking..."):
            result = chain_with_sources.invoke(user_question)
            response_text = result['response']
        
        # Stream the response for better UX
        streamed_response = stream_response(response_text)
        
        # Display source information if available
        if 'sources' in result and result['sources']:
            st.markdown("**Sources:**")
            source_text = ""
            for source in result['sources']:
                doc_name = source.get("document", "Unknown")
                page_num = source.get("page", "Unknown")
                source_text += f"- {doc_name} (Page: {page_num})\n"
            st.markdown(source_text)
        
        # Display images if any
        displayed_images = []
        if result['context']['images']:
            st.markdown("**Referenced Images:**")
            # Create columns for multiple images
            if len(result['context']['images']) > 1:
                img_cols = st.columns(min(len(result['context']['images']), 3))
                for i, image in enumerate(result['context']['images']):
                    with img_cols[i % 3]:
                        img_data = display_base64_image(image)
                        st.image(img_data, use_container_width=True)
                        displayed_images.append(img_data)
            else:
                for image in result['context']['images']:
                    img_data = display_base64_image(image)
                    st.image(img_data)
                    displayed_images.append(img_data)
        
        # Add to conversation history with sources
        sources_list = []
        if 'sources' in result and result['sources']:
            sources_list = result['sources']
            
        st.session_state.messages.append({
            "role": "assistant", 
            "content": streamed_response,
            "images": displayed_images if displayed_images else [],
            "sources": sources_list
        })

# Add app statistics and info to sidebar
if st.session_state.files_processed and st.session_state.all_chunks:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Document Statistics")
    text_chunks = sum(1 for chunk in st.session_state.all_chunks if "CompositeElement" in str(type(chunk)))
    table_chunks = sum(1 for chunk in st.session_state.all_chunks if "Table" in str(type(chunk)))
    
    # Get images from the chunks
    if st.session_state.rag_system:
        images = st.session_state.rag_system.get_images_base64(st.session_state.all_chunks)
    else:
        images = []
        
    stat_cols = st.sidebar.columns(3)
    stat_cols[0].metric("📝 Text Chunks", text_chunks)
    stat_cols[1].metric("📊 Tables", table_chunks)
    stat_cols[2].metric("🖼️ Images", len(images))

# Add app info
st.sidebar.markdown("---")
st.sidebar.info(
    "📚 **About This App**  \n"
    "This app uses LangChain and OpenAI to process documents and answer questions. "
    "Upload your PDFs, process them, and then ask questions about their content.  \n\n"
    "💡 **Tip**: For best results, ask specific questions about the document content."
)

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='text-align: center; color: #888;'>"
    "Made with Streamlit, LangChain & OpenAI"
    "</div>", 
    unsafe_allow_html=True
)