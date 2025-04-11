"""
MultimodalRAG is a class that can be used to process PDF files and
add them to a vectorstore. It can also be used to create a question
answering chain that can be used to answer questions based on the
content of the PDF files.

Parameters
----------
persist_directory : str
    The directory where the vectorstore will be stored.
"""
# Standard library imports
import os
import pickle
import re
import uuid
from base64 import b64decode
from typing import List, Tuple

# Third-party imports
from unstructured.partition.pdf import partition_pdf

# LangChain imports
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class PersistentInMemoryStore(InMemoryStore):
    """A wrapper around InMemoryStore that persists data to disk.
    
    This class extends InMemoryStore to add persistence by saving
    documents to a pickle file on disk.
    
    Parameters
    ----------
    persist_path : str
        Path to the file where the store will be persisted.
    """
    
    def __init__(self, persist_path: str):
        """Initialize the PersistentInMemoryStore.
        
        Parameters
        ----------
        persist_path : str
            Path to the file where the store will be persisted.
        """
        super().__init__()
        self.persist_path = persist_path
        # Ensure the directory exists
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        # Dictionary to store documents for persistence
        self.doc_dict = {}
        self._load()
    
    def _load(self):
        """Load the store from disk if it exists."""
        if os.path.exists(self.persist_path) and os.path.getsize(self.persist_path) > 0:
            try:
                with open(self.persist_path, 'rb') as f:
                    self.doc_dict = pickle.load(f)
                    # Load documents into the InMemoryStore
                    items = [(k, v) for k, v in self.doc_dict.items()]
                    if items:
                        super().mset(items)
                    print(f"Loaded {len(items)} documents from {self.persist_path}")
            except Exception as e:
                print(f"Error loading store from {self.persist_path}: {e}")
    
    def _save(self):
        """Save the store to disk."""
        try:
            # Create a new file to avoid corruption
            temp_path = f"{self.persist_path}.tmp"
            with open(temp_path, 'wb') as f:
                pickle.dump(self.doc_dict, f)
                f.flush()
                os.fsync(f.fileno())
            
            # Rename the temp file to the actual file (atomic operation)
            os.replace(temp_path, self.persist_path)
            print(f"Saved {len(self.doc_dict)} documents to {self.persist_path}")
        except Exception as e:
            print(f"Error saving store to {self.persist_path}: {e}")
    
    def mset(self, key_value_pairs):
        """Store multiple documents and persist to disk.
        
        Parameters
        ----------
        key_value_pairs : List[Tuple[str, Document]]
            List of (document_id, document) pairs to store.
        """
        # Update our local dictionary first
        for key, value in key_value_pairs:
            self.doc_dict[key] = value
        
        # Then update the parent store
        super().mset(key_value_pairs)
        
        # Save to disk
        self._save()
    
    def get(self, key):
        """Get a document by its ID.
        
        Parameters
        ----------
        key : str
            Document ID to retrieve.
            
        Returns
        -------
        Document
            The retrieved document.
        """
        # First try to get from the parent store
        results = super().mget([key])
        if results and len(results) > 0:
            return results[0]
        
        # If not found, check our local dictionary
        if key in self.doc_dict:
            return self.doc_dict[key]
            
        return None
    
    def delete(self, key):
        """Delete a document by its ID and persist changes.
        
        Parameters
        ----------
        key : str
            Document ID to delete.
        """
        # Remove from our local dictionary
        if key in self.doc_dict:
            del self.doc_dict[key]
        
        # Remove from parent store
        super().delete(key)
        
        # Save changes
        self._save()

class MultimodalRAG:
    def __init__(self, persist_directory="database"):
        """
        Initialize the MultimodalRAG system.

        Parameters
        ----------
        persist_directory : str
            The directory where the vectorstore will be stored.
        """
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vectorstore = Chroma(
            collection_name="document",
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )
        
        # Storage layer for parent documents - persistent storage
        self.id_key = "doc_id"
        
        # Create persistent store with the same directory as vectorstore
        persist_path = os.path.join(persist_directory, "docstore.pkl")
        self.store = PersistentInMemoryStore(persist_path)
        
        # Initialize retriever
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            id_key=self.id_key,
        )
        
    def get_images_base64(self, chunks, min_width=200, min_height=200, max_signature_ratio=5.0):
        """
        Extract base64 encoded images from chunks, filtering out small images,
        logos, headers, footers, and digital signatures.

        Parameters
        ----------
        chunks : List[unstructured.InlineChunk]
            List of chunks to extract images from.
        min_width : int, optional
            Minimum width in pixels for an image to be included, by default 200
        min_height : int, optional
            Minimum height in pixels for an image to be included, by default 200
        max_signature_ratio : float, optional
            Maximum width-to-height ratio for potential signature images, by default 5.0

        Returns
        -------
        List[str]
            List of filtered base64 encoded images.
        """
        import io
        from PIL import Image
        import base64

        images_b64 = []
        page_heights = {}  # Track page heights to identify headers/footers
        
        # First pass: collect page dimensions
        for chunk in chunks:
            if "CompositeElement" in str(type(chunk)):
                chunk_els = chunk.metadata.orig_elements
                for el in chunk_els:
                    if "Image" in str(type(el)):
                        try:
                            # Get page number if available
                            page_num = getattr(el.metadata, 'page_number', None)
                            if page_num is not None and hasattr(el.metadata, 'page_height'):
                                page_heights[page_num] = getattr(el.metadata, 'page_height', 1000)
                        except Exception:
                            pass
        
        # Second pass: filter images
        for chunk in chunks:
            if "CompositeElement" in str(type(chunk)):
                chunk_els = chunk.metadata.orig_elements
                for el in chunk_els:
                    if "Image" in str(type(el)):
                        try:
                            # Get image data
                            img_data = el.metadata.image_base64
                            
                            # Decode base64 and get image dimensions
                            img_bytes = base64.b64decode(img_data)
                            img = Image.open(io.BytesIO(img_bytes))
                            width, height = img.size
                            
                            # Skip small images (likely logos or icons)
                            if width < min_width or height < min_height:
                                print(f"Filtered out small image: {width}x{height}")
                                continue
                                
                            # Skip images with extreme aspect ratios (likely signatures or decorative elements)
                            aspect_ratio = max(width / height, height / width)
                            if aspect_ratio > max_signature_ratio:
                                print(f"Filtered out potential signature: aspect ratio {aspect_ratio:.2f}")
                                continue
                            
                            # Skip images at top/bottom of page (likely headers/footers)
                            page_num = getattr(el.metadata, 'page_number', None)
                            if page_num is not None and page_num in page_heights:
                                y_coord = getattr(el.metadata, 'y0', None)
                                page_height = page_heights[page_num]
                                
                                if y_coord is not None:
                                    # Check if image is at top 10% or bottom 10% of page
                                    if y_coord < 0.1 * page_height or y_coord > 0.9 * page_height:
                                        print(f"Filtered out header/footer image at position {y_coord/page_height:.2f}")
                                        continue
                            
                            # If we got here, the image passed all filters
                            images_b64.append(img_data)
                        except Exception as e:
                            # If there's an error processing the image, skip it
                            print(f"Error processing image: {str(e)}")
                            continue
        return images_b64
    
    def is_cover_page(self, chunk) -> bool:
        """
        Detect if a chunk is likely from a cover page.
        
        Parameters
        ----------
        chunk : unstructured.InlineChunk
            The chunk to analyze
        
        Returns
        -------
        bool
            True if the chunk is likely from a cover page, False otherwise
        """
        # Check if this is from the first page
        page_num = getattr(chunk.metadata, 'page_number', None)
        # Only consider the first page as a potential cover page
        if page_num != 1:
            return False
        
        # Get the text content
        text = str(chunk)
        text_lower = text.lower()
        
        # Cover page indicators
        cover_indicators = [
            # Common document types that appear on cover pages
            "white paper", "technical report", "case study", "user manual",
            "guide", "handbook", "confidential", "draft", "final report",
            # Copyright and legal text common on cover pages
            "all rights reserved", "copyright ", "proprietary",
            # Author/organization patterns
            "prepared by", "reviewed by", "approved by", "published by",
            # Amendment patterns
            "revision", "revision no.", "amended", "amended by", "amendment record sheet", 
        ]
        
        # Check for cover page indicators
        indicator_matches = sum(1 for indicator in cover_indicators if indicator in text_lower)
        
        # Check if the chunk is short (cover pages often have limited text)
        is_short = len(text) < 500
        
        # Check for date patterns (common on cover pages)
        date_patterns = [r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b']
        has_date = any(re.search(pattern, text) for pattern in date_patterns)
        
        # Return True if multiple indicators are found
        return (indicator_matches >= 2) or (indicator_matches >= 1 and (is_short or has_date))

    def is_table_of_contents(self, chunk) -> bool:
        """
        Detect if a chunk is likely from a table of contents.
        
        Parameters
        ----------
        chunk : unstructured.InlineChunk
            The chunk to analyze
        
        Returns
        -------
        bool
            True if the chunk is likely from a table of contents, False otherwise
        """
        # Check if this is from the first 3 pages (TOCs are typically on pages 2-3)
        page_num = getattr(chunk.metadata, 'page_number', None)
        if page_num is None or page_num > 3:
            return False
            
        # Get the text content
        text = str(chunk)
        text_lower = text.lower()
        
        # Check for TOC headers
        toc_headers = ["table of contents"]
        has_toc_header = any(header in text_lower for header in toc_headers)
        
        # Check for page number patterns (common in TOCs)
        # Look for patterns like "Section Name...........23" or "Chapter 1: Introduction 5"
        page_number_patterns = [
            r'\.\.+\s*\d+',  # Dots followed by numbers
            r'\s{2,}\d+$',  # Multiple spaces followed by numbers at end of line
            r'\b\d+\s*$'    # Numbers at the end of a line
        ]
        
        # Split by newlines to analyze line by line
        lines = text.split('\n')
        
        # Count lines with page number patterns
        pattern_matches = 0
        for line in lines:
            if any(re.search(pattern, line) for pattern in page_number_patterns):
                pattern_matches += 1
        
        # If we have a TOC header or multiple lines with page numbers, it's likely a TOC
        return has_toc_header or (pattern_matches >= 3 and pattern_matches / len(lines) >= 0.3)

    def process_pdf(self, file_path: str, original_filename: str = None, remove_cover_toc: bool = True) -> Tuple[List, List, List, List]:
        """
        Process a PDF file and extract chunks.

        Parameters
        ----------
        file_path : str
            Path to the PDF file to process.
        original_filename : str, optional
            Original filename of the document, by default None
        remove_cover_toc : bool, optional
            Whether to remove cover page and table of contents, by default True

        Returns
        -------
        Tuple[List, List, List, List]
            Tuple of lists containing:
            - chunks: List of extracted chunks
            - texts: List of extracted texts
            - tables: List of extracted tables
            - images: List of extracted images
        """
        # Extract content from PDF
        chunks = partition_pdf(
            filename=file_path,
            infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
            chunking_strategy="by_title",
            max_characters=12000,
            combine_text_under_n_chars=2000,
            new_after_n_chars=6000,
        )
        
        # Add source file metadata
        for chunk in chunks:
            if not hasattr(chunk.metadata, 'source'):
                # Use original filename if provided, otherwise use the temp filename
                source_name = original_filename if original_filename else os.path.basename(file_path)
                chunk.metadata.source = source_name
        
        # Filter out cover page and table of contents if requested
        if remove_cover_toc:
            filtered_chunks = []
            for chunk in chunks:
                # Skip chunks from cover page or table of contents
                if self.is_cover_page(chunk) or self.is_table_of_contents(chunk):
                    # Log what we're removing for debugging purposes
                    chunk_type = "cover page" if self.is_cover_page(chunk) else "table of contents"
                    page_num = getattr(chunk.metadata, 'page_number', 'unknown')
                    print(f"Removing {chunk_type} content from page {page_num}")
                    continue
                filtered_chunks.append(chunk)
            chunks = filtered_chunks
        
        # Separate elements into tables, text, and images
        tables = []
        texts = []
        
        for chunk in chunks:
            if "Table" in str(type(chunk)):
                tables.append(chunk)
            if "CompositeElement" in str(type((chunk))):
                texts.append(chunk)
        
        # Get images
        images = self.get_images_base64(chunks)
        
        return chunks, texts, tables, images
    
    def create_summaries(self, texts, tables, images, model_name="gpt-4o-mini"):
        """
        Create summaries for texts, tables, and images.

        Parameters
        ----------
        texts : List[Document]
            List of text chunks.
        tables : List[Document]
            List of table chunks.
        images : List[str]
            List of base64 encoded images.
        model_name : str, optional
            Name of the model to use for summarization, by default "gpt-4o-mini"

        Returns
        -------
        Tuple[List[str], List[str], List[str]]
            Tuple of lists containing:
            - text_summaries: List of text summaries
            - table_summaries: List of table summaries
            - image_summaries: List of image summaries
        """
        # Text and table summary prompt
        prompt_text = """
        You are an assistant tasked with summarizing tables and text.
        Give a concise summary of the table or text.
        
        Respond only with the summary, no additionnal comment.
        Do not start your message by saying "Here is a summary" or anything like that.
        Just give the summary as it is.
        
        Table or text chunk: {element}
        """
        prompt = ChatPromptTemplate.from_template(prompt_text)
        
        # Summary model
        summarize_model = ChatOpenAI(temperature=0.5, model=model_name)
        summarize_chain = {"element": lambda x: x} | prompt | summarize_model | StrOutputParser()
        
        # Process text summaries
        text_summaries = []
        if texts:
            text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})
        
        # Process table summaries
        table_summaries = []
        if tables:
            tables_html = [table.metadata.text_as_html for table in tables]
            table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3})
        
        # Process image summaries
        image_summaries = []
        if images:
            prompt_template = """Describe the image in detail. For context,
                            the image is part of a document. Be specific about graphs, diagrams, or visual elements."""
            messages = [
                (
                    "user",
                    [
                        {"type": "text", "text": prompt_template},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,{image}"},
                        },
                    ],
                )
            ]
            
            prompt = ChatPromptTemplate.from_messages(messages)
            chain = prompt | ChatOpenAI(model=model_name) | StrOutputParser()
            
            image_summaries = chain.batch(images)
        
        return text_summaries, table_summaries, image_summaries
    
    def add_to_vectorstore(self, texts, tables, images, text_summaries, table_summaries, image_summaries):
        """
        Add documents and summaries to the vectorstore.

        Parameters
        ----------
        texts : List[Document]
            List of text chunks.
        tables : List[Document]
            List of table chunks.
        images : List[str]
            List of base64 encoded images.
        text_summaries : List[str]
            List of text summaries.
        table_summaries : List[str]
            List of table summaries.
        image_summaries : List[str]
            List of image summaries.
        """
        # Add texts
        if texts and text_summaries:
            doc_ids = [str(uuid.uuid4()) for _ in texts]
            summary_texts = [
                Document(page_content=summary, metadata={self.id_key: doc_ids[i]}) 
                for i, summary in enumerate(text_summaries)
            ]
            self.retriever.vectorstore.add_documents(summary_texts)
            self.retriever.docstore.mset(list(zip(doc_ids, texts)))
        
        # Add tables
        if tables and table_summaries:
            table_ids = [str(uuid.uuid4()) for _ in tables]
            summary_tables = [
                Document(page_content=summary, metadata={self.id_key: table_ids[i]}) 
                for i, summary in enumerate(table_summaries)
            ]
            self.retriever.vectorstore.add_documents(summary_tables)
            self.retriever.docstore.mset(list(zip(table_ids, tables)))
        
        # Add image summaries
        if images and image_summaries:
            img_ids = [str(uuid.uuid4()) for _ in images]
            summary_img = [
                Document(page_content=summary, metadata={self.id_key: img_ids[i]}) 
                for i, summary in enumerate(image_summaries)
            ]
            self.retriever.vectorstore.add_documents(summary_img)
            self.retriever.docstore.mset(list(zip(img_ids, images)))
    
    def process_files(self, file_paths, original_filenames=None, summary_model="gpt-4o-mini"):
        """
        Process multiple files and add to the vectorstore.

        Parameters
        ----------
        file_paths : List[str]
            List of paths to PDF files to process.
        original_filenames : Dict[str, str], optional
            Mapping of temporary file paths to original filenames, by default None
        summary_model : str, optional
            Name of the model to use for summarization, by default "gpt-4o-mini"

        Returns
        -------
        List[Document]
            List of processed chunks.
        """
        all_chunks = []
        all_texts = []
        all_tables = []
        all_images = []
        
        # Process each file
        for file_path in file_paths:
            # Get original filename if available
            original_filename = None
            if original_filenames and file_path in original_filenames:
                original_filename = original_filenames[file_path]
                
            chunks, texts, tables, images = self.process_pdf(file_path, original_filename)
            all_chunks.extend(chunks)
            all_texts.extend(texts)
            all_tables.extend(tables)
            all_images.extend(images)
        
        # Create summaries
        text_summaries, table_summaries, image_summaries = self.create_summaries(
            all_texts, all_tables, all_images, model_name=summary_model
        )
        
        # Add to vectorstore
        self.add_to_vectorstore(
            all_texts, all_tables, all_images, 
            text_summaries, table_summaries, image_summaries
        )
        
        return all_chunks
    
    def parse_docs(self, docs):
        """
        Split base64-encoded images and texts.

        Parameters
        ----------
        docs : List[str]
            List of base64 encoded strings.

        Returns
        -------
        Dict[str, List[str]]
            Dictionary containing:
            - images: List of base64 encoded images
            - texts: List of base64 encoded texts
        """
        b64 = []
        text = []
        for doc in docs:
            try:
                b64decode(doc)
                b64.append(doc)
            except Exception:
                text.append(doc)
        return {"images": b64, "texts": text}
    
    def build_prompt(self, kwargs):
        """Build prompt with context for the LLM with improved structure and guidance.
        
        Args:
            kwargs: Dictionary containing context and question.
                context: Dictionary of retrieved documents by type (texts, images).
                question: The user's question to answer.
                
        Returns:
            ChatPromptTemplate: A formatted prompt template with structured context.
        """
        docs_by_type = kwargs["context"]
        user_question = kwargs["question"]
    
        # Build structured text context with clear separations
        context_text = ""
        if len(docs_by_type["texts"]) > 0:
            for i, text_element in enumerate(docs_by_type["texts"]):
                # Add document separators with basic numbering
                # Safely access metadata if it exists
                doc_id = f"Document {i+1}"
                page_num = "Unknown"
                
                # Try to access metadata attributes directly if they exist
                if hasattr(text_element, "metadata"):
                    metadata = text_element.metadata
                    if hasattr(metadata, "doc_id"):
                        doc_id = metadata.doc_id
                    if hasattr(metadata, "page"):
                        page_num = metadata.page
                
                context_text += f"\n--- TEXT SEGMENT {i+1} (Source: {doc_id}, Page: {page_num}) ---\n"
                context_text += text_element.text + "\n"
    
        # Construct improved prompt with better instructions and context framing
        prompt_template = f"""
        Answer the question based only on the following context, which includes text, tables, and images from documents.
        
        CONTEXT:
        {context_text}
        
        INSTRUCTIONS:
        1. Use only information provided in the context above and images below
        2. If the answer cannot be determined from the context, say "I cannot answer this based on the provided information"
        3. Provide concise, factual answers without speculation
        4. For information from tables or images, explicitly mention this in your answer
        
        QUESTION: {user_question}
        """
    
        prompt_content = [{"type": "text", "text": prompt_template}]
    
        # Add images with descriptive labels
        if len(docs_by_type["images"]) > 0:
            prompt_content.append({"type": "text", "text": "\nREFERENCE IMAGES:"})
            for i, image in enumerate(docs_by_type["images"]):
                # Add image caption if available in metadata
                prompt_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    }
                )
                # Add a separator after each image for clarity
                if i < len(docs_by_type["images"]) - 1:
                    prompt_content.append({"type": "text", "text": "\n---\n"})
    
        return ChatPromptTemplate.from_messages(
            [
                HumanMessage(content=prompt_content),
            ]
        )
    
    def get_qa_chain(self, model_name="gpt-4o"):
        """Create a question-answering chain."""
        chain = (
            {
                "context": self.retriever | RunnableLambda(self.parse_docs),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(self.build_prompt)
            | ChatOpenAI(model=model_name)
            | StrOutputParser()
        )
        return chain
    
    def get_qa_chain_with_sources(self, model_name="gpt-4o"):
        """Create a question-answering chain that returns sources.
        
        Returns a chain that provides the response text along with source information
        including document names and page numbers.
        
        Args:
            model_name: The name of the OpenAI model to use
            
        Returns:
            A runnable chain that returns a dictionary with response text and source metadata
        """
        # Extract source metadata from documents
        def extract_source_metadata(docs_dict):
            sources = []
            for doc in docs_dict.get("texts", []):
                if hasattr(doc, "metadata"):
                    source_info = {
                        "document": getattr(doc.metadata, "source", "Unknown"),
                        "page": getattr(doc.metadata, "page_number", "Unknown")
                    }
                    # Only add unique sources
                    if source_info not in sources:
                        sources.append(source_info)
            return sources
            
        chain_with_sources = {
            "context": self.retriever | RunnableLambda(self.parse_docs),
            "question": RunnablePassthrough(),
        } | RunnablePassthrough().assign(
            response=(
                RunnableLambda(self.build_prompt)
                | ChatOpenAI(model=model_name)
                | StrOutputParser()
            ),
            sources=lambda x: extract_source_metadata(x["context"])
        )
        return chain_with_sources