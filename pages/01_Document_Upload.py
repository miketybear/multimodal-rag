import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
import sys

# Add parent directory to path to import from parent module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

# Initialize session state variables if not already set
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
if "rag_system" not in st.session_state:
    # Initialize RAG system if database exists
    if check_database_exists():
        st.session_state.rag_system = MultimodalRAG(persist_directory="database")
        st.session_state.files_processed = True
    else:
        st.session_state.rag_system = None
        st.session_state.files_processed = False
# Add processing status flag
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

# Callback function to process documents
def process_documents_callback():
    st.session_state.is_processing = True

# Page header with custom styling
st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='color: #4257f5;'>📋 Document Upload</h1>
        <p style='font-size: 1.2em; color: #666;'>Upload and process your documents for Q&A</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Function to process uploaded files
def process_uploaded_files(uploaded_files, model_name):
    """Process uploaded PDF files and build the retriever"""
    # Initialize the RAG system
    rag_system = MultimodalRAG(persist_directory="database")
    
    # Save uploaded files to temp files and process
    temp_file_paths = []
    original_filenames = {}
    
    try:
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_paths.append(temp_file.name)
                # Store mapping of temp file to original filename
                original_filenames[temp_file.name] = uploaded_file.name
        
        # Process all files with original filenames
        all_chunks = rag_system.process_files(
            temp_file_paths, 
            original_filenames=original_filenames,
            summary_model="gpt-4o-mini"
        )
        
        return rag_system, all_chunks
    finally:
        # Clean up temporary files
        for temp_path in temp_file_paths:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

# Model selection
st.markdown("### 🤖 Model Settings")
model_name = st.selectbox(
    "Select OpenAI Model",
    ["gpt-4o", "gpt-4o-mini"],
    index=1  # Default to gpt-4o-mini as it's cheaper
)

# Document upload section
st.markdown("### 📄 Upload Documents")
uploaded_files = st.file_uploader(
    "Upload PDF documents", 
    type=['pdf'], 
    accept_multiple_files=True,
    help="Upload one or more PDF files to analyze"
)

process_button = st.button(
    "📥 Process Documents", 
    use_container_width=True, 
    disabled=st.session_state.is_processing,
    on_click=process_documents_callback if uploaded_files else None
)

# Process documents when processing flag is set
if st.session_state.is_processing and uploaded_files:
    # Process documents with a spinner
    with st.spinner("Processing documents..."):
        st.session_state.rag_system, st.session_state.all_chunks = process_uploaded_files(uploaded_files, model_name)
        st.session_state.files_processed = True
    
    # Reset processing flag
    st.session_state.is_processing = False
    
    st.success(f"✅ Successfully processed {len(uploaded_files)} documents with {len(st.session_state.all_chunks)} chunks!")
    
    # Display document details
    st.subheader("Document Summary")
    
    # Count elements by type
    text_chunks = sum(1 for chunk in st.session_state.all_chunks if "CompositeElement" in str(type(chunk)))
    table_chunks = sum(1 for chunk in st.session_state.all_chunks if "Table" in str(type(chunk)))
    
    # Get images from the chunks
    if st.session_state.rag_system:
        images = st.session_state.rag_system.get_images_base64(st.session_state.all_chunks)
    else:
        images = []
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📝 Text Chunks", text_chunks)
    with col2:
        st.metric("📊 Tables", table_chunks)
    with col3:
        st.metric("🖼️ Images", len(images))
        
    # Add a link to the chat page
    st.markdown("---")
    st.markdown("### 🚀 Next Steps")
    st.info("Your documents have been processed! Go to the Chat page to start asking questions about your documents.")
    
elif not uploaded_files:
    st.info("Please upload PDF documents to begin.")
    
elif not process_button:
    st.info("Click 'Process Documents' to analyze the uploaded PDF files.")

# Add app info at the bottom
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
    <p>This app uses LangChain and OpenAI to process documents and answer questions.</p>
    <p>Upload your PDFs, process them, and then ask questions about their content.</p>
    </div>
    """, 
    unsafe_allow_html=True
)
