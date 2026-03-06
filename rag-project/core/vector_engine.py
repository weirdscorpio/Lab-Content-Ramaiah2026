"""
Vector Engine - Manages vector database operations using ChromaDB
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import numpy as np
from datetime import datetime
import os

class VectorEngine:
    def __init__(self, collection_name: str = "docs"):
        print("[VectorEngine] Initializing ChromaDB...")
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embedding model
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '0'  # Allow downloading
        
        print("[VectorEngine] Loading embedding model...")
        print("[VectorEngine] Note: First-time download may take 2-3 minutes (~90MB)")
        
        try:
            # Try to load model with progress indication
            import sys
            sys.stdout.flush()
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            print("[VectorEngine] Model loaded successfully!")
        except Exception as e:
            print(f"[VectorEngine] Error loading model: {e}")
            print("[VectorEngine] Attempting offline mode...")
            # Try to work offline if model exists
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            import warnings
            warnings.filterwarnings('ignore')
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.last_updated = datetime.now()
    
    def add_document(self, doc_id: str, text: str, metadata: Dict[str, Any]):
        """Add a document chunk to the vector store"""
        # Generate embedding
        embedding = self.embedding_model.encode(text).tolist()
        
        # Add to collection
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata]
        )
        
        self.last_updated = datetime.now()
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents using semantic similarity"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'title': results['metadatas'][0][i].get('title', 'Unknown'),
                    'snippet': results['documents'][0][i][:150] + '...'
                })
        
        return formatted_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        count = self.collection.count()
        
        # Count unique documents
        all_metadata = self.collection.get()['metadatas']
        unique_files = set()
        if all_metadata:
            for meta in all_metadata:
                unique_files.add(meta.get('file', ''))
        
        return {
            'total_chunks': count,
            'total_documents': len(unique_files),
            'last_updated': self.last_updated.isoformat()
        }
    
    def clear_collection(self):
        """Clear all documents from the collection"""
        # Delete and recreate collection
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
        self.last_updated = datetime.now()
    
    def is_initialized(self) -> bool:
        """Check if the vector store has been initialized with documents"""
        return self.collection.count() > 0
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text (useful for visualization)"""
        return self.embedding_model.encode(text).tolist()
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """Get data for vector space visualization"""
        # This is a simplified version - in practice, you'd use dimensionality reduction
        try:
            # Get a sample of documents
            sample = self.collection.get(limit=50, include=["embeddings", "metadatas"])
            
            if not sample['embeddings']:
                return {"error": "No documents in collection"}
            
            # Simple 2D projection using first two dimensions
            points = []
            for i, embedding in enumerate(sample['embeddings']):
                points.append({
                    'x': embedding[0],
                    'y': embedding[1],
                    'category': sample['metadatas'][i].get('category', 'unknown'),
                    'title': sample['metadatas'][i].get('title', 'Unknown')
                })
            
            return {
                'points': points,
                'categories': list(set(p['category'] for p in points))
            }
        except Exception as e:
            return {"error": str(e)}
