"""
Tests for the PDF processing functionality in the MultimodalRAG class.
"""
import os
import pytest
from unittest.mock import MagicMock, patch

from multimodal_rag import MultimodalRAG


class TestPDFProcessing:
    """Test cases for PDF processing functionality."""

    def setup_method(self):
        """Set up test environment before each test method."""
        self.rag = MultimodalRAG(persist_directory="test_database")
        
    def teardown_method(self):
        """Clean up test environment after each test method."""
        # Clean up test database if it exists
        if os.path.exists("test_database"):
            import shutil
            shutil.rmtree("test_database")

    def test_is_cover_page_positive(self):
        """Test that cover pages are correctly identified."""
        # Create a mock chunk with cover page characteristics
        mock_chunk = MagicMock()
        mock_chunk.metadata.page_number = 1
        mock_chunk.__str__.return_value = """
        TECHNICAL REPORT
        
        Project Analysis and Recommendations
        
        Prepared by: Data Science Team
        January 15, 2024
        
        CONFIDENTIAL
        """
        
        # Test the function
        result = self.rag.is_cover_page(mock_chunk)
        assert result is True, "Should identify text as a cover page"

    def test_is_cover_page_negative(self):
        """Test that non-cover pages are correctly identified."""
        # Create a mock chunk with non-cover page characteristics
        mock_chunk = MagicMock()
        mock_chunk.metadata.page_number = 2  # Not first page
        mock_chunk.__str__.return_value = """
        Chapter 1: Introduction
        
        This document describes the methodology used in our analysis.
        The project began in early 2023 with a comprehensive review of
        existing literature and industry best practices.
        """
        
        # Test the function
        result = self.rag.is_cover_page(mock_chunk)
        assert result is False, "Should not identify text as a cover page"

    def test_is_table_of_contents_positive(self):
        """Test that table of contents pages are correctly identified."""
        # Create a mock chunk with table of contents characteristics
        mock_chunk = MagicMock()
        # Set page number to 2 (typical for TOC)
        mock_chunk.metadata.page_number = 2
        mock_chunk.__str__.return_value = """
        TABLE OF CONTENTS
        
        Executive Summary..........................3
        Introduction...............................5
        Methodology................................7
        Results....................................12
        Discussion.................................18
        Conclusion.................................25
        References.................................28
        """
        
        # Test the function
        result = self.rag.is_table_of_contents(mock_chunk)
        assert result is True, "Should identify text as a table of contents"

    def test_is_table_of_contents_negative(self):
        """Test that non-table of contents pages are correctly identified."""
        # Create a mock chunk with non-table of contents characteristics
        mock_chunk = MagicMock()
        # Set page number to 4 (beyond our TOC check range)
        mock_chunk.metadata.page_number = 4
        mock_chunk.__str__.return_value = """
        Chapter 2: Methodology
        
        Our research methodology consisted of three phases:
        1. Data collection
        2. Analysis
        3. Validation
        
        Each phase was conducted according to industry standards.
        """
        
        # Test the function
        result = self.rag.is_table_of_contents(mock_chunk)
        assert result is False, "Should not identify text as a table of contents"

    def test_process_pdf_removes_cover_and_toc(self):
        """Test that process_pdf removes cover page and table of contents when requested."""
        # Create mock chunks
        cover_chunk = MagicMock()
        cover_chunk.metadata.page_number = 1
        
        toc_chunk = MagicMock()
        toc_chunk.metadata.page_number = 2
        
        content_chunk = MagicMock()
        content_chunk.metadata.page_number = 3
        
        # Set up metadata for chunks
        for chunk in [cover_chunk, toc_chunk, content_chunk]:
            chunk.metadata.source = None
            chunk.metadata.orig_elements = []
        
        # Create patches
        with patch('multimodal_rag.partition_pdf', return_value=[cover_chunk, toc_chunk, content_chunk]), \
             patch.object(self.rag, 'get_images_base64', return_value=[]), \
             patch.object(self.rag, 'is_cover_page', side_effect=lambda x: x == cover_chunk), \
             patch.object(self.rag, 'is_table_of_contents', side_effect=lambda x: x == toc_chunk), \
             patch('multimodal_rag.str') as mock_str:
            
            # Set up str type checking for the "Table" and "CompositeElement" checks
            def mock_str_side_effect(obj):
                if obj == type(cover_chunk) or obj == type(toc_chunk) or obj == type(content_chunk):
                    return "CompositeElement"
                return str(obj)
            
            mock_str.side_effect = mock_str_side_effect
            
            # Call process_pdf with remove_cover_toc=True
            chunks, texts, tables, images = self.rag.process_pdf(
                file_path="dummy.pdf",
                original_filename="original.pdf",
                remove_cover_toc=True
            )
            
            # Verify that only the content chunk remains
            assert len(chunks) == 1, "Should have filtered out cover and TOC chunks"
            assert chunks[0] == content_chunk, "The remaining chunk should be the content chunk"
            
            # Verify that the texts list contains only the content chunk
            assert len(texts) == 1, "Should have filtered out cover and TOC chunks from texts"
            assert texts[0] == content_chunk, "The remaining text should be the content chunk"

    def test_process_pdf_keeps_cover_and_toc_when_requested(self):
        """Test that process_pdf keeps cover page and table of contents when requested."""
        # Create mock chunks (same as previous test)
        cover_chunk = MagicMock()
        cover_chunk.metadata.page_number = 1
        
        toc_chunk = MagicMock()
        toc_chunk.metadata.page_number = 2
        
        content_chunk = MagicMock()
        content_chunk.metadata.page_number = 3
        
        # Set up metadata for chunks
        for chunk in [cover_chunk, toc_chunk, content_chunk]:
            chunk.metadata.source = None
            chunk.metadata.orig_elements = []
        
        # Create patches - similar to previous test but with remove_cover_toc=False
        with patch('multimodal_rag.partition_pdf', return_value=[cover_chunk, toc_chunk, content_chunk]), \
             patch.object(self.rag, 'get_images_base64', return_value=[]), \
             patch.object(self.rag, 'is_cover_page', return_value=False), \
             patch.object(self.rag, 'is_table_of_contents', return_value=False), \
             patch('multimodal_rag.str') as mock_str:
            
            # Set up str type checking for the "Table" and "CompositeElement" checks
            def mock_str_side_effect(obj):
                if obj == type(cover_chunk) or obj == type(toc_chunk) or obj == type(content_chunk):
                    return "CompositeElement"
                return str(obj)
            
            mock_str.side_effect = mock_str_side_effect
            
            # Call process_pdf with remove_cover_toc=False
            chunks, texts, tables, images = self.rag.process_pdf(
                file_path="dummy.pdf",
                original_filename="original.pdf",
                remove_cover_toc=False
            )
            
            # Verify that all chunks remain
            assert len(chunks) == 3, "Should keep all chunks when remove_cover_toc=False"
            assert cover_chunk in chunks, "Cover chunk should be included"
            assert toc_chunk in chunks, "TOC chunk should be included"
            assert content_chunk in chunks, "Content chunk should be included"
            
            # Verify that the texts list contains all chunks
            assert len(texts) == 3, "Should keep all text chunks when remove_cover_toc=False"
