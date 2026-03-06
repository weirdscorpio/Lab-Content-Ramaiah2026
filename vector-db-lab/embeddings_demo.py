#!/usr/bin/env python3
"""
Using sentence-transformers to show how text becomes meaningful vectors
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

def visualize_embedding_slice(embedding, start=0, end=10):
    """Show a small slice of the embedding vector visually"""
    print(f"\n📊 Showing dimensions {start}-{end} of 384 total dimensions:")
    print("  Value:  ", end="")
    for val in embedding[start:end]:
        # Show positive/negative with different symbols
        if val > 0.5:
            print("▲▲", end=" ")  # Strong positive
        elif val > 0:
            print("▲ ", end=" ")  # Positive
        elif val < -0.5:
            print("▼▼", end=" ")  # Strong negative
        else:
            print("▼ ", end=" ")  # Negative
    print()
    print("  Number: ", end="")
    for val in embedding[start:end]:
        print(f"{val:+.1f}", end=" ")
    print()

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def main():
    print("=" * 70)
    print("=" * 70)
    print("\nLet's use a real AI model (all-MiniLM-L6-v2) to convert text to vectors!")
    print("This model creates 384-dimensional vectors that capture meaning.\n")
    
    print("Loading AI model (this takes a few seconds)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model loaded!\n")
    
    input("➡️  Press Enter to see the magic of embeddings...")
    
    # STEP 1: Show that similar words create similar embeddings
    print("\n" + "=" * 70)
    print("STEP 1: SIMILAR WORDS → SIMILAR NUMBERS")
    print("=" * 70)
    
    word_pairs = [
        ("holiday", "vacation"),          # From narration - key example!
        ("dress code", "clothing rules"),
        ("jeans", "denim"),
        ("Friday", "weekend")
    ]
    
    print("\nLet's compare similar concepts:")
    print("-" * 50)
    
    for word1, word2 in word_pairs:
        emb1 = model.encode(word1)
        emb2 = model.encode(word2)
        similarity = cosine_similarity(emb1, emb2)
        
        print(f'\n📝 "{word1}" vs "{word2}"')
        print(f"   Similarity: {similarity:.1%}")
        
        # Show a tiny slice of the vectors
        print(f'   "{word1}" vector (first 5 values): [{", ".join(f"{v:.2f}" for v in emb1[:5])}...]')
        print(f'   "{word2}" vector (first 5 values): [{", ".join(f"{v:.2f}" for v in emb2[:5])}...]')
    
    print("\n💡 Notice how 'holiday' and 'vacation' - completely different words -")
    print("   have high similarity because the AI understands they mean similar things!")
    
    # NEW: Add dimensionality explanation
    print("\n" + "=" * 70)
    print("📏 WHY 384 DIMENSIONS?")
    print("=" * 70)
    print("""
Why do we need so many dimensions? Each dimension captures different aspects:
• Tone (formal vs casual)
• Context (work vs personal)
• Formality level
• Semantic category
• And hundreds more subtle features!

Think of it like describing a person:
• 1D: Just height (not enough!)
• 2D: Height + weight (better)
• 384D: Everything from personality to preferences (complete picture!)

Industry standard: 768-1536 dimensions for production systems.
Our model (all-MiniLM-L6-v2): 384 dimensions - perfect balance!
""")
    
    input("\n➡️  Press Enter to see the burden shift...")
    
    # NEW: Add burden shift discussion
    print("\n" + "=" * 70)
    print("⚖️ THE TRADEOFF: WHO DOES THE WORK?")
    print("=" * 70)
    print("""
SQL Database:
❌ Burden on the SEARCHER
   • Must know exact keywords
   • Must try multiple variations
   • "clothing" won't find "dress code"
   
Vector Database:
✅ Burden on the SETUP (that's you!)
   • You configure embeddings
   • You set similarity thresholds
   • You handle chunking
   
But the result?
🎯 Natural language "just works" for users!
   • "What should I wear?" finds dress code policy
   • "Holiday time off" finds vacation policy
   • No training needed for users!
""")
    
    input("\n➡️  Press Enter to solve search problem...")
    
    # STEP 2: Solve the original search problem
    print("\n" + "=" * 70)
    print("STEP 2: SOLVING THE SEARCH PROBLEM FROM THE ORIGINAL SEARCH PROBLEM")
    print("=" * 70)
    
    # The policy from our database
    policy = "Business casual attire required Monday through Thursday. Jeans permitted on Fridays."
    policy_embedding = model.encode(policy)
    
    print(f'\n📄 Company Policy: "{policy}"')
    print(f"🔢 Converted to 384-dimensional vector")
    visualize_embedding_slice(policy_embedding, 0, 10)
    
    # Test queries with varying relevance
    queries = [
        "Can I wear jeans on Friday?",        # Very relevant - direct match
        "What's the dress code policy?",      # Relevant - about dress code
        "When can I take vacation?",          # Not relevant - different topic
        "How's the weather today?"            # Completely unrelated
    ]
    
    print("\n" + "=" * 50)
    print("Testing employee questions with varying relevance:")
    print("=" * 50)
    print("Watch how the AI accurately scores relevance!")
    
    for query in queries:
        query_embedding = model.encode(query)
        similarity = cosine_similarity(policy_embedding, query_embedding)
        
        print(f'\n❓ "{query}"')
        print(f"   Similarity to dress code policy: {similarity:.1%}")
        
        if similarity > 0.5:
            print("   ✅ STRONG MATCH - Highly relevant to dress code!")
        elif similarity > 0.3:
            print("   🔗 PARTIAL MATCH - Somewhat related")
        elif similarity > 0.2:
            print("   ⚠️  WEAK MATCH - Different topic")
        else:
            print("   ❌ NO MATCH - Completely unrelated")
    
    print("\n💡 Notice how the AI correctly identifies:")
    print("   • 'Jeans on Friday' → High similarity (directly mentioned in policy)")
    print("   • 'Dress code policy' → Good similarity (same topic)")
    print("   • 'Vacation' → Lower similarity (different HR topic)")
    print("   • 'Weather' → Very low similarity (completely unrelated)")
    
    input("\n➡️  Press Enter to try the interactive demo...")
    
    # STEP 3: Interactive comparison
    print("\n" + "=" * 70)
    print("🎮 INTERACTIVE: COMPARE ANY TWO TEXTS")
    print("=" * 70)
    print("\nType two texts and see how similar the AI thinks they are!")
    
    while True:
        text1 = input("\n📝 Enter first text: ")
        text2 = input("📝 Enter second text: ")
        
        if text1 and text2:
            emb1 = model.encode(text1)
            emb2 = model.encode(text2)
            similarity = cosine_similarity(emb1, emb2)
            
            print(f"\n🔍 Comparing:")
            print(f'   Text 1: "{text1}"')
            print(f'   Text 2: "{text2}"')
            print(f"\n📊 Results:")
            print(f"   Similarity: {similarity:.1%}")
            
            if similarity > 0.8:
                print("   🎯 Very similar meaning!")
            elif similarity > 0.5:
                print("   ✅ Related topics")
            elif similarity > 0.3:
                print("   🔗 Somewhat related")
            else:
                print("   ❌ Different topics")
            
            # Show vector comparison
            print("\nVector comparison (first 10 dimensions):")
            print("Text 1:", end="")
            visualize_embedding_slice(emb1, 0, 10)
            print("Text 2:", end="")
            visualize_embedding_slice(emb2, 0, 10)
        
        # Ask if they want to try another
        another = input("\n🔄 Would you like to try another comparison? (y/n): ")
        if another.lower() not in ['y', 'yes']:
            break
        
    # Save completion marker
    with open('embeddings_demo.txt', 'w') as f:
        f.write("Model used: all-MiniLM-L6-v2 (384 dimensions)\n")
        f.write("Key Learning: Real embeddings capture semantic meaning\n")
    
if __name__ == "__main__":
    main()