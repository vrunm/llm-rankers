import os
from haystack import Document
from listwise import ListwiseRanker  

def test_listwise_ranker():
    # Configuration - use environment variables in production
    GROQ_API_KEY = "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"
    MODEL_NAME = "mixtral-8x7b-32768"
    
    # More distinct test documents for better ranking differentiation
    documents = [
        Document(
            content="Association football, commonly known as soccer, is a team sport played between two teams of 11 players.",
            meta={"source": "wiki", "year": 2023}
        ),
        Document(
            content="American football is a sport played by two teams of 11 players on a rectangular field with goalposts at each end.",
            meta={"source": "textbook", "year": 2020}
        ),
        Document(
            content="Tennis is a racket sport played individually against a single opponent or between two teams of two players each.",
            meta={"source": "sports_db", "year": 2022}
        ),
        Document(
            content="The history of football extends back to ancient times with various traditional games involving kicking balls.",
            meta={"source": "history_archive", "year": 2021}
        )
    ]
    
    # Initialize the ranker component
    reranker = ListwiseRanker(
        model_name=MODEL_NAME,
        api_key=GROQ_API_KEY,
        window_size=3,
        step_size=2,
        num_repeat=1
    )
    
    # Test query with clear intent
    query = "Explain the rules of association football (soccer)"
    
    try:
        # Execute ranking
        result = reranker.run(query=query, documents=documents)
        reranked_docs = result["documents"]
        
        # Display input/output comparison
        print("\nOriginal documents:")
        for i, doc in enumerate(documents):
            print(f"[{i+1}] {doc.meta['source']}: {doc.content[:45]}...")
            
        print("\nReranked results:")
        for i, doc in enumerate(reranked_docs):
            print(f"[{i+1}] Score: {doc.score} | {doc.meta['source']}: {doc.content[:45]}...")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Troubleshooting:")
        print("- Verify GROQ_API_KEY is valid")
        print("- Check network connection")
        print("- Confirm package versions: pip install groq haystack")

if __name__ == "__main__":
    test_listwise_ranker()
