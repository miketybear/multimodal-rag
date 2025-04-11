# Multimodal RAG System

A Retrieval Augmented Generation (RAG) system that processes PDF documents containing text, tables, and images. This system uses LangChain, OpenAI models, and Chroma vector database to create a document Q&A system with a modern Streamlit interface.

## Features

- **PDF Processing**: Extract text, tables, and images from PDF documents
- **Multimodal Content Handling**: Process and retrieve information from different content types
- **Persistent Storage**: Document storage persists between application sessions using custom PersistentInMemoryStore
- **Interactive Q&A Interface**: Ask questions about your documents through a modern Streamlit web interface
- **Streaming Responses**: Improved UX with text streaming for more responsive interactions
- **Image Display**: View images referenced in answers for better context
- **Document Statistics**: Track the number of text chunks, tables, and images in your documents

## Architecture

The system consists of the following components:

1. **Document Processing**: Uses Unstructured to extract content from PDFs
2. **Vector Storage**: Chroma vector database for semantic search
3. **Document Storage**: Custom PersistentInMemoryStore for document persistence
4. **Retrieval System**: MultiVectorRetriever for efficient multimodal content retrieval
5. **LLM Integration**: OpenAI models (GPT-4o and GPT-4o-mini) for generating answers
6. **Web Interface**: Modern Streamlit UI with multi-page navigation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multimodal-rag.git
cd multimodal-rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key in a `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run streamlit_app.py
```

2. The application has two main pages:
   - **Chat with DocuBot**: Ask questions about your processed documents
   - **Document Upload**: Upload and process PDF documents

3. Workflow:
   - First, go to the Document Upload page to upload and process your PDF files
   - Then, switch to the Chat page to ask questions about your documents
   - The system will retrieve relevant content and generate answers using OpenAI models

## Project Structure

- `multimodal_rag.py`: Core RAG implementation with persistent document storage
- `streamlit_app.py`: Main Streamlit application with navigation
- `pages/`: Streamlit pages
  - `01_Document_Upload.py`: Interface for uploading and processing documents
  - `02_Chat.py`: Chat interface for asking questions about documents
- `tests/`: Unit tests for the system
- `requirements.txt`: Project dependencies
- `database/`: Directory for storing vector database and document store

## Persistent Document Storage

The system uses a custom `PersistentInMemoryStore` class that extends LangChain's `InMemoryStore` to provide document persistence between application sessions. This implementation:

1. Maintains a local dictionary of documents that's persisted to disk using pickle
2. Uses atomic file operations to prevent data corruption
3. Implements proper document retrieval across different application instances
4. Ensures documents are preserved between application sessions

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Model Options

The application supports two OpenAI models:
- **GPT-4o**: Higher quality responses but more expensive
- **GPT-4o-mini**: Faster and more cost-effective for most use cases

## Docker Deployment

The application can be containerized using Docker for easy deployment and consistency across environments.

### Building the Docker Image

```bash
# Navigate to the project directory
cd multimodal-rag

# Build the Docker image
docker build -t multimodal-rag:latest .
```

### Running the Docker Container

```bash
# Run the container, mapping port 8555 and mounting a volume for persistent storage
docker run -p 8555:8555 \
  -v $(pwd)/database:/app/database \
  -e OPENAI_API_KEY=your_api_key_here \
  --name multimodal-rag-app \
  multimodal-rag:latest
```

### Accessing the Application

Once the container is running, access the application in your browser at:
```
http://localhost:8555
```

### Docker Compose (Optional)

For easier management, you can create a `docker-compose.yml` file:

```yaml
version: '3'

services:
  multimodal-rag:
    build: .
    ports:
      - "8555:8555"
    volumes:
      - ./database:/app/database
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
```

And run with:
```bash
docker-compose up -d
```

## License

[MIT License](LICENSE)
