"""
Document Processor - Handles chunking and processing of TechCorp documents
"""

import os
import re
from typing import List, Dict, Any
from pathlib import Path
import hashlib

class DocumentProcessor:
    def __init__(self, vector_engine):
        self.vector_engine = vector_engine
        self.docs_path = Path("../docs")
        
        # Chunking parameters
        self.chunk_size = 500  # characters
        self.chunk_overlap = 100  # characters
        
    def process_all_documents(self) -> Dict[str, int]:
        """Process all documents in the TechCorp docs folder"""
        processed_count = 0
        chunk_count = 0
        
        # Clear existing data
        self.vector_engine.clear_collection()
        
        # Process markdown files directly in the docs folder
        for doc_file in self.docs_path.glob("*.md"):
            print(f"\nProcessing document: {doc_file.name}")
            chunks = self.process_document(doc_file, "doc_file.name")
            chunk_count += len(chunks)
            processed_count += 1
        
        print(f"\nTotal processed: {processed_count} documents, {chunk_count} chunks")
        return {
            "processed": processed_count,
            "chunks": chunk_count
        }
    
    def process_document(self, file_path: Path, category: str) -> List[Dict[str, Any]]:
        """Process a single document into chunks"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract metadata from document
        title = self._extract_title(content)
        doc_id = hashlib.md5(str(file_path).encode()).hexdigest()
        
        # Create chunks
        chunks = self._create_chunks(content)
        
        # Process each chunk
        processed_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk_data = {
                "id": f"{doc_id}_{i}",
                "text": chunk_text,
                "metadata": {
                    "title": title or file_path.stem.replace('-', ' ').title(),
                    "category": category,
                    "file": file_path.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            }
            
            # Add to vector store
            self.vector_engine.add_document(
                chunk_data["id"],
                chunk_text,
                chunk_data["metadata"]
            )
            
            processed_chunks.append(chunk_data)
        
        return processed_chunks
    
    def _extract_title(self, content: str) -> str:
        """Extract title from markdown document"""
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        return match.group(1) if match else None
    
    def _create_chunks(self, text: str) -> List[str]:
        """Create overlapping chunks from text"""
        chunks = []
        
        # Simple chunking by character count with overlap
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to find a good break point (end of sentence or paragraph)
            if end < len(text):
                # Look for sentence end
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + self.chunk_size // 2:
                    end = sentence_end + 1
                else:
                    # Look for paragraph break
                    para_break = text.rfind('\n\n', start, end)
                    if para_break > start + self.chunk_size // 2:
                        end = para_break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            
        return chunks
    
    def chunk_text_by_strategy(self, text: str, strategy: str = "paragraph") -> List[str]:
        """Chunk text using different strategies for lab exercises"""
        if strategy == "fixed":
            return self._create_chunks(text)
        
        elif strategy == "sentence":
            # Split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= self.chunk_size:
                    current_chunk += " " + sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return chunks
        
        elif strategy == "paragraph":
            # Split by paragraphs
            paragraphs = text.split('\n\n')
            chunks = []
            
            for para in paragraphs:
                if len(para) <= self.chunk_size:
                    chunks.append(para.strip())
                else:
                    # Split large paragraphs
                    sub_chunks = self._create_chunks(para)
                    chunks.extend(sub_chunks)
            
            return chunks
        
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")