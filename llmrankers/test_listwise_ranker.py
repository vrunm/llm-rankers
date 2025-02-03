import os
from haystack import Document
from listwise import GroqRankLLMReranker  # Replace with actual module name

def test_groq_reranker():
    # Configuration
    GROQ_API_KEY = "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"
    MODEL_NAME = "mixtral-8x7b-32768"
    
    # Sample documents (should be related to test query)
    documents = [
        Document(
            content="Football",
            meta={"source": "wiki"}
        ),
        Document(
            content="Football",
            meta={"source": "textbook"}
        ),
        Document(
            content="Tennis",
            meta={"source": "research_paper"}
        ),
        Document(
            content="Cricket",
            meta={"source": "tech_journal"}
        )
    ]
    
    # Create reranker component
    reranker = GroqRankLLMReranker(
        model_name=MODEL_NAME,
        api_key=GROQ_API_KEY,
        window_size=3,
        step_size=2,
        num_repeat=1
    )
    print(reranker)
    # Test query
    query = "What is football?"
    
    try:
        # Run reranking
        result = reranker.run(query=query, documents=documents)
        reranked_docs = result["documents"]
        
        # Display results
        print("\nOriginal order:")
        for i, doc in enumerate(documents):
            print(f"[{i+1}] {doc.content[:50]}...")
            
        print("\nReranked order:")
        for i, doc in enumerate(reranked_docs):
            print(f"[{i+1}] Score: {doc.score} | {doc.content[:50]}...")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Ensure you have:")
        print("1. A valid Groq API key")
        print("2. Installed required packages: pip install groq tiktoken")
        print("3. Internet connection")

if __name__ == "__main__":
    test_groq_reranker()
