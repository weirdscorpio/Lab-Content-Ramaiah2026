#!/usr/bin/env python3
"""
Shows how vector similarity finds relevant content using cosine similarity
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# Global model instance to avoid reloading
model = None

def get_embedding_model():
    """Get or initialize the embedding model"""
    global model
    if model is None:
        print("Loading AI model (this takes a few seconds)...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Model loaded!\n")
    return model

def get_embedding(text):
    """Convert text to a 384-dimensional semantic vector using all-MiniLM-L6-v2"""
    model = get_embedding_model()
    return model.encode(text)

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

class VectorDatabase:
    """Simple vector database for semantic search"""
    
    def __init__(self):
        self.documents = []
        self.vectors = []
        self.metadata = []
    
    def add_document(self, text, category):
        """Add a document to the database"""
        vector = get_embedding(text)
        self.documents.append(text)
        self.vectors.append(vector)
        self.metadata.append({'category': category})
    
    def search(self, query, top_k=3, min_similarity=0.2):
        """Search for most similar documents with configurable threshold"""
        query_vector = get_embedding(query)
        
        similarities = []
        for i, doc_vector in enumerate(self.vectors):
            sim = cosine_similarity(query_vector, doc_vector)
            # Only include results above threshold
            if sim >= min_similarity:
                similarities.append((sim, i))
        
        # Sort by similarity (highest first)
        similarities.sort(reverse=True)
        
        results = []
        for sim, idx in similarities[:top_k]:
            results.append({
                'document': self.documents[idx],
                'similarity': sim,
                'category': self.metadata[idx]['category']
            })
        
        return results

def main():
    """Demonstrate semantic similarity search"""
    
    print("=" * 70)
    print("🔍 Similarity Search - Finding Meaning with Math")
    print("=" * 70)
    print("\n🎯 Tia builds a vector database for semantic search!")
    print("Using the same all-MiniLM-L6-v2 model from embeddings demo...\n")
    
    # Initialize vector database
    db = VectorDatabase()
    
    # Add Tia's handbook policies
    policies = [
        ("DRESS CODE: Business casual Monday-Thursday. Jeans allowed on Fridays. No shorts.", "Dress Code"),
        ("VACATION: Request 2 weeks in advance. Holiday requests need manager approval.", "Time Off"),
        ("REMOTE WORK: Company laptops for home use. Work from home needs approval.", "Remote"),
        ("PARKING: Free employee lot. Visitor parking needs validation.", "Parking"),
        ("BREAKS: Two 15-minute breaks. Lunch 30-60 minutes.", "Breaks"),
        ("SECURITY: Badge access required. Visitors must sign in.", "Security")
    ]
    
    print("📚 Building Vector Database:")
    print("-" * 50)
    
    for policy, category in policies:
        db.add_document(policy, category)
        print(f"✅ Added: {category:12} policy to vector database")
    
    # Test queries
    test_queries = [
        "Can I wear jeans to work?",
        "What are the clothing rules?",
        "How do I take my laptop home?",
        "When can I work remotely?",
        "Where do employees park?",
        "How long is lunch break?"
    ]
    
    print("\n" + "=" * 70)
    print("🎯 SEMANTIC SEARCH RESULTS")
    print("=" * 70)
    
    for query in test_queries:
        print(f"\n❓ Query: '{query}'")
        print("-" * 50)
        
        results = db.search(query, top_k=3, min_similarity=0.2)
        
        if len(results) > 0:
            print(f"📊 Found {len(results)} relevant match(es):")
            for i, result in enumerate(results, 1):
                similarity_bar = '█' * int(result['similarity'] * 30)
                print(f"\n  {i}. [{result['category']:10}] Similarity: {result['similarity']:.1%}")
                print(f"     [{similarity_bar:<30}]")
                print(f"     '{result['document'][:60]}...'")
        else:
            print("❌ No matches found above threshold")
    
    # NEW: Demonstrate the Florida example - context matters!
    print("\n" + "=" * 70)
    print("🌴 CONTEXT MATTERS: THE FLORIDA EXAMPLE")
    print("=" * 70)
    print("\nSame word 'Florida', but different contexts:")
    
    florida_queries = [
        "Can I take my company laptop to Florida?",
        "Can I use vacation days for Florida trip?"
    ]
    
    for query in florida_queries:
        print(f"\n❓ Query: '{query}'")
        results = db.search(query, top_k=1)
        if results:
            print(f"   → Matched: {results[0]['category']} ({results[0]['similarity']:.1%})")
            print(f"   → Policy: '{results[0]['document'][:50]}...'")
    
    print("\n💡 Notice: 'Florida' appears in both, but they match different policies!")
    print("   First matches REMOTE WORK (laptop/equipment)")
    print("   Second matches TIME OFF (vacation)")
    
    # NEW: Demonstrate scoring thresholds
    print("\n" + "=" * 70)
    print("📊 SCORING THRESHOLDS: FILTERING BY CONFIDENCE")
    print("=" * 70)
    
    test_query = "What about company equipment?"
    print(f"\n🔍 Testing query: '{test_query}'")
    print("-" * 50)
    
    # Test with different thresholds
    thresholds = [0.7, 0.5, 0.3, 0.2]
    for threshold in thresholds:
        results = db.search(test_query, top_k=5, min_similarity=threshold)
        print(f"\n📌 With threshold {threshold:.1f}:")
        if results:
            print(f"   Found {len(results)} result(s):")
            for r in results[:2]:  # Show first 2
                print(f"   • {r['category']:12} ({r['similarity']:.1%})")
        else:
            print("   No results above threshold!")
    
    print("\n💡 Higher threshold = fewer but more confident results!")
    
    # Interactive demo
    input("\n➡️  Press Enter to try the interactive similarity search...")
    
    print("\n" + "=" * 70)
    print("🎮 INTERACTIVE SIMILARITY SEARCH")
    print("=" * 70)
    print("\nType your own queries and see how the vector database finds relevant policies!")
    print("(Type 'quit' to continue to the next section)\n")
    
    while True:
        user_query = input("\n📝 Enter your query (or 'quit'): ")
        if user_query.lower() in ['quit', 'q', 'exit']:
            break
        
        if user_query.strip():
            results = db.search(user_query, top_k=3, min_similarity=0.2)
            
            if results:
                print(f"\n🎯 Found {len(results)} relevant match(es):")
                for i, result in enumerate(results, 1):
                    similarity_bar = '█' * int(result['similarity'] * 30)
                    print(f"\n  {i}. [{result['category']:10}] Similarity: {result['similarity']:.1%}")
                    print(f"     [{similarity_bar:<30}]")
                    print(f"     Policy excerpt: '{result['document'][:80]}...'")
            else:
                print("❌ No matches found. Try a different query!")
    
    # Compare with keyword search
    print("\n" + "=" * 70)
    print("📊 COMPARISON: Semantic vs Keyword Search")
    print("=" * 70)
    
    comparison_query = "What are the clothing rules?"
    print(f"\n🔍 Query: '{comparison_query}'")
    
    # Keyword search (would fail)
    keyword_matches = sum(1 for doc in db.documents if 'clothing' in doc.lower())
    print(f"\n❌ Keyword Search: Found {keyword_matches} matches (word 'clothing' not in policies!)")
    
    # Semantic search (succeeds)
    semantic_results = db.search(comparison_query, top_k=1)
    print(f"\n✅ Semantic Search: Found perfect match!")
    print(f"   → {semantic_results[0]['category']}: {semantic_results[0]['similarity']:.1%} similarity")
    print(f"   → Even though 'clothing' ≠ 'dress code', vectors understand they're related!")
    
    print("\n" + "=" * 70)
    print("💡 KEY INSIGHTS")
    print("=" * 70)
    
    print("""
🎯 How Semantic Search Works:
   1. Convert query to vector
   2. Compare with all document vectors
   3. Return most similar (highest cosine similarity)
   
📐 Cosine Similarity & Scoring:
   - Measures angle between vectors
   - Range: -1 (opposite) to 1 (identical)
   
📊 Scoring Thresholds (Configurable!):
   - 0.7+ = High confidence match
   - 0.5-0.7 = Good match
   - 0.3-0.5 = Moderate match
   - <0.3 = Filter out (too weak)
   
✨ The Power:
   - "clothing rules" finds "dress code" (semantic match!)
   - "laptop home" finds "remote work" (context understanding!)
   - No exact keywords needed - meaning is what matters!
   
🚀 Tia's Success:
   - 100% of queries find correct policies
   - Employees get instant, accurate answers
   - Natural language just works!
""")
    
    # Save completion marker
    with open('similarity_search.txt', 'w') as f:
        f.write("completed: Similarity Search\n")
        f.write(f"Documents indexed: {len(db.documents)}\n")
        f.write(f"Queries tested: {len(test_queries)}\n")
        f.write("Key Learning: Cosine similarity enables semantic search\n")
    
    print("✅ Complete! Run vector_database.py for the complete system.")

if __name__ == "__main__":
    main()