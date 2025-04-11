# -*- coding: utf-8 -*-
"""
langchain-multimodal-improved.py

Improved version of the multimodal RAG script using LangChain, Chroma,
and unstructured, incorporating suggestions for clarity and robustness.
"""

import base64
import uuid
import os
import logging
from typing import List, Dict, Any, Union

# --- Dependencies Installation (Run these in your environment/notebook) ---
# %pip install -Uq "unstructured[all-docs]" pillow lxml beautifulsoup4 # Added beautifulsoup4 for better HTML parsing if needed
# %pip install -Uq chromadb tiktoken
# %pip install -Uq langchain langchain_community langchain_openai langchain_groq
# %pip install -Uq python_dotenv ipython # IPython for display in notebooks

# --- Library Imports ---
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Element, Table, CompositeElement, Image as UnstructuredImage

from IPython.display import Image as IPythonImage, display

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables (replace with your actual keys or ensure .env file exists)
load_dotenv()
# Check if essential keys are loaded
if not os.getenv("OPENAI_API_KEY"):
    logging.warning("OPENAI_API_KEY not found in environment variables.")
# Add checks for GROQ_API_KEY, LANGCHAIN_API_KEY if you use those services

# File and Directory Paths
INPUT_DIR = "input/"
PDF_FILENAME = 'IT_Policy_BDPOC.pdf' # Make sure this file exists in INPUT_DIR
PDF_FILE_PATH = os.path.join(INPUT_DIR, PDF_FILENAME)
CHROMA_PERSIST_DIR = "database_improved"
CHROMA_COLLECTION_NAME = "multimodal_rag_doc"

# Model Names
EMBEDDING_MODEL = "text-embedding-3-large"
SUMMARY_MODEL = "gpt-4o-mini" # Or use "gpt-4o", or a Groq model if configured
FINAL_RAG_MODEL = "gpt-4o-mini"

# Unstructured Partitioning Settings
PARTITION_STRATEGY = "hi_res" # "hi_res" needed for table extraction
MAX_CHARS = 4000 # Reduced max chars for potentially smaller chunks
NEW_AFTER_N_CHARS = 3800 # Slightly less than max_chars
COMBINE_UNDER_N_CHARS = 500 # Combine small elements

# Retriever Settings
ID_KEY = "doc_id"

# --- Helper Functions ---

def display_base64_image(base64_string: str):
    """Decodes base64 string and displays the image in IPython."""
    try:
        image_data = base64.b64decode(base64_string)
        display(IPythonImage(data=image_data))
    except Exception as e:
        logging.error(f"Error displaying base64 image: {e}")

def looks_like_base64(s: Any) -> bool:
    """Check if a string looks like base64."""
    if not isinstance(s, str):
        return False
    try:
        # Attempt to decode, adjusting for padding issues if necessary
        padding = '=' * (-len(s) % 4)
        base64.b64decode(s + padding)
        # Optionally add more checks (e.g., length, character set)
        return True
    except (base64.binascii.Error, ValueError, TypeError):
        return False

def parse_retrieved_docs(docs: List[Union[Element, Document, str]]) -> Dict[str, List[Any]]:
    """
    Parses retrieved documents from the docstore into structured context data.
    Separates images (as base64) from text/table elements.
    Handles cases where raw strings might be stored (e.g., if image storage failed).
    """
    image_b64_list = []
    text_table_elements = []

    for doc in docs:
        if isinstance(doc, UnstructuredImage):
            if hasattr(doc.metadata, 'image_base64') and doc.metadata.image_base64:
                image_b64_list.append(doc.metadata.image_base64)
            # Add fallback if you store raw bytes?
            # elif hasattr(doc, 'image_bytes'):
            #     img_b64 = base64.b64encode(doc.image_bytes).decode()
            #     image_b64_list.append(img_b64)
            else:
                logging.warning(f"Retrieved Image element has no base64 data: ID {getattr(doc.metadata, ID_KEY, 'Unknown')}")
        elif isinstance(doc, (Table, CompositeElement)): # Add other relevant text types if needed
            text_table_elements.append(doc)
        elif isinstance(doc, Document) and looks_like_base64(doc.page_content):
             # Handle case where base64 string was stored directly in a Document wrapper perhaps
             logging.warning("Retrieved a Document that looks like base64, attempting to treat as image.")
             image_b64_list.append(doc.page_content)
        elif isinstance(doc, Document):
             # Assume it's text-based content wrapped in LangChain Document
             logging.info(f"Retrieved generic Document, treating as text: {doc.metadata}")
             # Create a minimal compatible structure or extract text
             # This part might need adjustment based on exactly WHAT gets stored if not an unstructured Element
             text_table_elements.append(doc) # Or extract doc.page_content into a simpler format
        elif isinstance(doc, str) and looks_like_base64(doc):
            # Handle case where just a raw base64 string was stored (original approach)
            logging.warning("Retrieved a raw string that looks like base64, treating as image.")
            image_b64_list.append(doc)
        elif isinstance(doc, str):
             logging.warning(f"Retrieved a raw string, treating as text: {doc[:100]}...")
             # Wrap in a simple object or Document if needed by build_prompt
             text_table_elements.append(Document(page_content=doc)) # Example wrapping
        else:
            logging.warning(f"Unexpected document type retrieved from docstore: {type(doc)}")

    return {"images": image_b64_list, "texts": text_table_elements}


def build_rag_prompt(kwargs: Dict[str, Any]) -> ChatPromptTemplate:
    """Builds the multimodal prompt for the RAG chain."""
    retrieved_docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_parts = []
    page_numbers = set()

    if retrieved_docs_by_type["texts"]:
        context_parts.append("--- Relevant Text and Table Context ---")
        for element in retrieved_docs_by_type["texts"]:
            page_num = None
            # Try getting page number from metadata
            if hasattr(element, 'metadata') and hasattr(element.metadata, 'page_number'):
                 page_num = element.metadata.page_number
                 if page_num:
                     page_numbers.add(page_num)

            page_info = f"(Page {page_num})" if page_num else ""

            if isinstance(element, Table):
                context_parts.append(f"\n[Table Context {page_info}]:")
                # Prefer HTML for structure, fallback to text
                content = getattr(element.metadata, 'text_as_html', element.text)
                context_parts.append(content)
            elif hasattr(element, 'text'): # For CompositeElement, NarrativeText etc.
                context_parts.append(f"\n[Text Context {page_info}]:")
                context_parts.append(element.text)
            elif isinstance(element, Document): # Handle fallback Document object
                 context_parts.append(f"\n[Context {page_info}]:")
                 context_parts.append(element.page_content)

    context_text = "\n".join(context_parts)
    page_num_str = f" (source pages: {', '.join(map(str, sorted(list(page_numbers))))})" if page_numbers else ""

    # Base prompt template
    prompt_template_text = f"""You are an assistant for question-answering tasks.
Answer the following question based *only* on the provided context{page_num_str}.
The context may include text, tables, and images. Be precise and concise.

Question: {user_question}

Context:
{context_text}
"""

    # Prepare message content list
    prompt_content: List[Union[Dict[str, Any], str]] = [{"type": "text", "text": prompt_template_text}]

    # Add images if they exist
    if retrieved_docs_by_type["images"]:
        # Add text instructing the model about the images BEFORE the images
        prompt_content[0]["text"] += "\n\n--- Relevant Image Context ---\n(The following images are part of the context)"

        for img_b64 in retrieved_docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}, # Assuming JPEG, might need adjustment
                }
            )
    else:
         prompt_content[0]["text"] += "\n\n(No relevant images were retrieved for this question.)"


    return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])


# --- Main Execution Logic ---
if __name__ == "__main__":

    # --- 1. Extract Data from PDF ---
    logging.info(f"Starting PDF partitioning for: {PDF_FILE_PATH}")
    if not os.path.exists(PDF_FILE_PATH):
        logging.error(f"PDF file not found at: {PDF_FILE_PATH}")
        exit()

    try:
        raw_pdf_elements: List[Element] = partition_pdf(
            filename=PDF_FILE_PATH,
            infer_table_structure=True,
            extract_images_in_pdf=False, # Deprecated, use extract_image_block_types
            strategy=PARTITION_STRATEGY,
            extract_image_block_types=["Image"], # Extract Image type elements
            extract_image_block_to_payload=True, # Embed base64 in metadata
            chunking_strategy="by_title", # Chunk semantically by titles
            max_characters=MAX_CHARS,
            new_after_n_chars=NEW_AFTER_N_CHARS,
            combine_text_under_n_chars=COMBINE_UNDER_N_CHARS,
            # image_output_dir_path=INPUT_DIR, # Uncomment to save images locally instead of payload
        )
        logging.info(f"Successfully partitioned PDF into {len(raw_pdf_elements)} elements.")
    except Exception as e:
        logging.error(f"Failed to partition PDF: {e}")
        exit()

    # Separate elements by type - Store the *full element object*
    text_elements: List[Element] = []
    table_elements: List[Table] = []
    image_elements: List[UnstructuredImage] = []

    for element in raw_pdf_elements:
        if isinstance(element, Table):
            table_elements.append(element)
        elif isinstance(element, CompositeElement):
            # CompositeElement often represents text sections under a title
            text_elements.append(element)
            # Check if images are nested within CompositeElement's metadata (can happen with 'by_title')
            if hasattr(element, 'metadata') and hasattr(element.metadata, 'orig_elements'):
                for el in element.metadata.orig_elements:
                     if isinstance(el, UnstructuredImage) and hasattr(el.metadata, 'image_base64'):
                         # Add nested images if they have base64 payload
                         image_elements.append(el)
                         logging.info(f"Found nested image within CompositeElement: {getattr(element, 'text', '')[:50]}...")
        elif isinstance(element, UnstructuredImage):
             if hasattr(element.metadata, 'image_base64') and element.metadata.image_base64:
                 image_elements.append(element)
             else:
                 logging.warning("Found Image element without base64 payload. Ensure extract_image_block_to_payload=True.")


    logging.info(f"Separated elements: {len(text_elements)} text sections, {len(table_elements)} tables, {len(image_elements)} images.")

    # Display first detected image if available (for verification)
    if image_elements:
        logging.info("Displaying the first extracted image:")
        first_image_b64 = image_elements[0].metadata.image_base64
        display_base64_image(first_image_b64)
    else:
        logging.info("No images with base64 payload were extracted.")

    # --- 2. Summarize Elements ---
    summary_llm = ChatOpenAI(temperature=0.2, model=SUMMARY_MODEL) # Lower temp for factual summaries

    # Summarize text elements
    text_summaries: List[str] = []
    if text_elements:
        logging.info("Summarizing text elements...")
        prompt_text = """You are an assistant tasked with summarizing text sections.
Give a concise summary of the text. Focus on the key information or purpose of the section.
Respond only with the summary, no additional comments like "Here is a summary:".
Text chunk: {element}"""
        prompt = ChatPromptTemplate.from_template(prompt_text)
        summarize_chain = {"element": lambda x: x.text} | prompt | summary_llm | StrOutputParser()
        try:
            text_summaries = summarize_chain.batch(text_elements, {"max_concurrency": 5})
            logging.info(f"Generated {len(text_summaries)} text summaries.")
        except Exception as e:
            logging.error(f"Failed to summarize text elements: {e}")

    # Summarize table elements
    table_summaries: List[str] = []
    if table_elements:
        logging.info("Summarizing table elements...")
        prompt_text = """You are an assistant tasked with summarizing tables.
Give a concise summary of the table's content and purpose. Mention key data points or structure if apparent.
Respond only with the summary, no additional comments like "Here is a summary:".
Table (HTML representation): {element}"""
        prompt = ChatPromptTemplate.from_template(prompt_text)
        # Use HTML representation for potentially better structure understanding by LLM
        summarize_chain = {"element": lambda x: x.metadata.text_as_html if hasattr(x.metadata, 'text_as_html') else x.text} | prompt | summary_llm | StrOutputParser()
        try:
            table_summaries = summarize_chain.batch(table_elements, {"max_concurrency": 5})
            logging.info(f"Generated {len(table_summaries)} table summaries.")
        except Exception as e:
            logging.error(f"Failed to summarize table elements: {e}")

    # Summarize image elements
    image_summaries: List[str] = []
    if image_elements:
        logging.info("Summarizing image elements...")
        img_summary_llm = ChatOpenAI(model=FINAL_RAG_MODEL, max_tokens=1024) # Use GPT-4o or similar for image understanding
        img_prompt_template = """Describe the image in detail. Explain its purpose or what it depicts in the context of a document.
If it's a diagram or chart, explain what it shows. If it's a picture, describe the scene."""
        img_messages_template = [
            HumanMessage(
                content=[
                    {"type": "text", "text": img_prompt_template},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{image_base64}"},
                    },
                ]
            )
        ]
        img_prompt = ChatPromptTemplate.from_messages(img_messages_template)
        img_summarize_chain = img_prompt | img_summary_llm | StrOutputParser()
        # Prepare batch input
        img_batch_input = [{"image_base64": img.metadata.image_base64} for img in image_elements]
        try:
            image_summaries = img_summarize_chain.batch(img_batch_input, {"max_concurrency": 3}) # Lower concurrency for multimodal model
            logging.info(f"Generated {len(image_summaries)} image summaries.")
            # print("First image summary:", image_summaries[0] if image_summaries else "N/A") # DEBUG
        except Exception as e:
            logging.error(f"Failed to summarize image elements: {e}")

    # Check if we have summaries to proceed
    if not text_summaries and not table_summaries and not image_summaries:
         logging.error("No summaries were generated. Exiting.")
         exit()

    # --- 3. Initialize Vector Store and Document Store ---
    logging.info("Initializing vector store and document store...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # Initialize Chroma, checking persistence
    if os.path.exists(CHROMA_PERSIST_DIR):
        logging.info(f"Loading existing Chroma vector store from: {CHROMA_PERSIST_DIR}")
        vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        # Simple approach: Assume if DB exists, it *might* be populated.
        # For a robust solution, you'd need to check counts or timestamps
        # and potentially clear/repopulate if the source PDF changed.
        # count = vectorstore._collection.count() # Check number of items
        # logging.info(f"Found {count} items in existing vector store collection.")
    else:
        logging.info(f"Creating new Chroma vector store at: {CHROMA_PERSIST_DIR}")
        vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        # Indicate that we need to add documents since it's new
        # (In this script's flow, we'll add regardless, but useful flag in complex apps)


    # The storage layer for the original elements (parent documents).
    # InMemoryStore is transient and will be rebuilt each run in this script.
    # For true persistence across runs, you'd need a persistent KeyValueStore
    # and logic to sync it with Chroma's state.
    store = InMemoryStore()

    # Initialize the MultiVectorRetriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=ID_KEY,
        # search_kwargs={'k': 5} # Optional: retrieve top 5 summaries
    )

    # --- 4. Add Data to Stores ---
    logging.info("Adding summaries to vector store and original elements to doc store...")

    # Helper to add documents and map to store
    def add_elements_to_retriever(elements: List[Element], summaries: List[str], retriever: MultiVectorRetriever):
        if not elements or not summaries or len(elements) != len(summaries):
            logging.warning(f"Skipping add for elements/summaries mismatch or empty lists. Elements: {len(elements)}, Summaries: {len(summaries)}")
            return
        element_ids = [str(uuid.uuid4()) for _ in elements]
        summary_docs = [
            Document(page_content=summary, metadata={ID_KEY: element_ids[i]})
            for i, summary in enumerate(summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        # Store the *original Unstructured element object* in the docstore
        retriever.docstore.mset(list(zip(element_ids, elements)))
        logging.info(f"Added {len(elements)} elements and their summaries.")

    # Add texts, tables, and images
    add_elements_to_retriever(text_elements, text_summaries, retriever)
    add_elements_to_retriever(table_elements, table_summaries, retriever)
    add_elements_to_retriever(image_elements, image_summaries, retriever)

    # Persist Chroma explicitly (though add_documents might do it depending on version/config)
    # vectorstore.persist() # Uncomment if needed, check Chroma docs for behavior

    # --- 5. Define RAG Chain ---
    logging.info("Defining the RAG chain...")

    # Use the refined helper functions defined earlier
    chain_with_sources = (
        {
            "context": retriever | RunnableLambda(parse_retrieved_docs),
            "question": RunnablePassthrough(),
        }
        | RunnablePassthrough().assign(
            response=(
                RunnableLambda(build_rag_prompt)
                | ChatOpenAI(model=FINAL_RAG_MODEL, temperature=0.1, max_tokens=1500) # Use powerful model for final answer
                | StrOutputParser()
            )
        )
    )

    # Simpler chain for just the response
    chain = (
        {
            "context": retriever | RunnableLambda(parse_retrieved_docs),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(build_rag_prompt)
        | ChatOpenAI(model=FINAL_RAG_MODEL, temperature=0.1, max_tokens=1500)
        | StrOutputParser()
    )


    # --- 6. Run Query ---
    logging.info("Running a sample query...")
    # query = "What is the policy on software installation?"
    # query = "Who authored this document?"
    query = "diễn giải quy trình quản lý tài khoản và email?" # Example Vietnamese query

    print("\n--- Query ---")
    print(query)

    # Invoke the chain that includes sources in the output
    try:
        result = chain_with_sources.invoke(query)

        print("\n--- Response ---")
        print(result.get('response', 'No response generated.'))

        print("\n--- Retrieved Context ---")
        retrieved_context = result.get('context', {})
        texts = retrieved_context.get('texts', [])
        images = retrieved_context.get('images', [])

        print(f"\nRetrieved {len(texts)} Text/Table Elements:")
        for i, text_el in enumerate(texts):
            page_num = getattr(text_el.metadata, 'page_number', 'N/A') if hasattr(text_el, 'metadata') else 'N/A'
            content_preview = getattr(text_el, 'text', getattr(text_el,'page_content','N/A'))[:200] # Handle both Element and Document
            print(f"{i+1}. Type: {type(text_el).__name__}, Page: {page_num}")
            print(f"   Preview: {content_preview}...")
            print("-" * 30)


        print(f"\nRetrieved {len(images)} Image Elements:")
        if images:
            for i, img_b64 in enumerate(images):
                print(f"Image {i+1}:")
                display_base64_image(img_b64)
        else:
            print("(No images retrieved)")

    except Exception as e:
        logging.error(f"Error invoking RAG chain: {e}")
        print(f"\nError processing query: {e}")

    logging.info("Script finished.")