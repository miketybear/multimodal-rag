"""
Unit tests for the persistent document storage in MultimodalRAG.
"""
import os
import shutil
import tempfile
import unittest
from typing import List, Dict, Any

from langchain.schema.document import Document
from multimodal_rag import MultimodalRAG, PersistentInMemoryStore


class TestPersistentStorage(unittest.TestCase):
    """Test cases for persistent document storage in MultimodalRAG."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for the database
        self.test_dir = tempfile.mkdtemp()
        self.rag = MultimodalRAG(persist_directory=self.test_dir)
        
        # Create a sample document for testing
        self.sample_doc = Document(
            page_content="This is a test document.",
            metadata={"source": "test", "page": 1}
        )
        
        # Sample document ID
        self.doc_id = "test_doc_id"
    
    def tearDown(self):
        """Clean up after each test."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_store_and_retrieve_document(self):
        """Test storing and retrieving a document."""
        # Store the document
        self.rag.store.mset([(self.doc_id, self.sample_doc)])
        
        # Retrieve the document
        retrieved_doc = self.rag.store.get(self.doc_id)
        
        # Check if the retrieved document matches the original
        self.assertEqual(retrieved_doc.page_content, self.sample_doc.page_content)
        self.assertEqual(retrieved_doc.metadata, self.sample_doc.metadata)
    
    def test_persistence_across_instances(self):
        """Test document persistence across different MultimodalRAG instances."""
        # Store a document using the first instance
        self.rag.store.mset([(self.doc_id, self.sample_doc)])
        
        # Print debug info
        print(f"Document stored with ID: {self.doc_id}")
        print(f"Store path: {self.rag.store.persist_path}")
        print(f"Directory contents: {os.listdir(self.test_dir)}")
        
        # Force sync to disk
        self.rag.store._save()
        
        # Create a new instance with the same persist_directory
        new_rag = MultimodalRAG(persist_directory=self.test_dir)
        
        # Print more debug info
        print(f"New store path: {new_rag.store.persist_path}")
        print(f"Directory contents after new instance: {os.listdir(self.test_dir)}")
        
        # Retrieve the document using the new instance
        retrieved_doc = new_rag.store.get(self.doc_id)
        
        # Debug retrieved document
        print(f"Retrieved document: {retrieved_doc}")
        
        # Check if the retrieved document matches the original
        self.assertIsNotNone(retrieved_doc, "Retrieved document should not be None")
        self.assertEqual(retrieved_doc.page_content, self.sample_doc.page_content)
        self.assertEqual(retrieved_doc.metadata, self.sample_doc.metadata)
    
    def test_multiple_documents(self):
        """Test storing and retrieving multiple documents."""
        # Create multiple documents
        docs = {
            "doc1": Document(page_content="Document 1", metadata={"source": "test1"}),
            "doc2": Document(page_content="Document 2", metadata={"source": "test2"}),
            "doc3": Document(page_content="Document 3", metadata={"source": "test3"})
        }
        
        # Store all documents
        items = [(doc_id, doc) for doc_id, doc in docs.items()]
        self.rag.store.mset(items)
        
        # Retrieve and verify each document
        for doc_id, original_doc in docs.items():
            retrieved_doc = self.rag.store.get(doc_id)
            self.assertEqual(retrieved_doc.page_content, original_doc.page_content)
            self.assertEqual(retrieved_doc.metadata, original_doc.metadata)
    
    def test_document_with_special_characters(self):
        """Test storing and retrieving documents with special characters."""
        # Create a document with special characters
        special_doc = Document(
            page_content="Special characters: !@#$%^&*()_+<>?:\"{}|~`-=[]\\;',./",
            metadata={"source": "special_test"}
        )
        special_id = "special_doc_id"
        
        # Store and retrieve the document
        self.rag.store.mset([(special_id, special_doc)])
        retrieved_doc = self.rag.store.get(special_id)
        
        # Check if the retrieved document matches the original
        self.assertEqual(retrieved_doc.page_content, special_doc.page_content)
        self.assertEqual(retrieved_doc.metadata, special_doc.metadata)


if __name__ == "__main__":
    unittest.main()
