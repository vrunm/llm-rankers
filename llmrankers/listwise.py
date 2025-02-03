from haystack import Document, component, default_from_dict, default_to_dict
from typing import List, Optional
import groq  
import tiktoken

def max_tokens(model):
    return 8192 if 'gpt-4' in model else 4096

def get_post_prompt(query, num):
    return f"Search Query: {query}. \nRank the {num} passages based on relevance. List in descending order using identifiers like [1] > [2]. Only provide ranking."

def get_prefix_prompt(query, num):
    return [
        {'role': 'system', 'content': "You are RankGPT, a relevance ranking assistant."},
        {'role': 'user', 'content': f"Rank {num} passages for query: {query}."},
        {'role': 'assistant', 'content': 'Ready for passages.'}
    ]

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    for message in messages:
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
    return num_tokens + 3

def create_permutation_instruction_chat(query: str, docs: List[Document], model_name=None):
    messages = get_prefix_prompt(query, len(docs))
    max_length = 300
    for i, doc in enumerate(docs):
        content = ' '.join(doc.content.replace('Title: Content: ', '').strip().split()[:max_length])
        messages.extend([
            {'role': 'user', 'content': f"[{i+1}] {content}"},
            {'role': 'assistant', 'content': f'Received passage [{i+1}]'}
        ])
    messages.append({'role': 'user', 'content': get_post_prompt(query, len(docs))})
    return messages



def clean_response(response: str):
    return ''.join(c if c.isdigit() else ' ' for c in response).strip()

def remove_duplicate(response):
    return sorted(set(response), key=response.index)

def receive_permutation(ranking, permutation, rank_start=0, rank_end=100):
    response = [int(x)-1 for x in clean_response(permutation).split()]
    response = remove_duplicate(response)
    cut_range = copy.deepcopy(ranking[rank_start:rank_end])
    return [cut_range[x] if x < len(cut_range) else x for x in response]

class GroqListwiseLlmRanker:
    def __init__(self, model_name: str, api_key: str, window_size=10, step_size=5, num_repeat=1):
        self.model_name = model_name
        self.window_size = window_size
        self.step_size = step_size
        self.num_repeat = num_repeat
        self.client = groq.Client(api_key=api_key)  # Groq client initialization

    def compare(self, query: str, docs: List[Document]):
        messages = create_permutation_instruction_chat(query, docs, self.model_name)
        
        # Groq API call
        completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            temperature=0.0
        )
        return completion.choices[0].message.content

    def rerank(self, query: str, ranking: List[Document]) -> List[Document]:
        # Same sliding window implementation using Haystack Documents
        for _ in range(self.num_repeat):
            ranking = copy.deepcopy(ranking)
            end_pos = len(ranking)
            while (start_pos := end_pos - self.window_size) >= 0:
                window_docs = ranking[start_pos:end_pos]
                result = self.compare(query, window_docs)
                ranking = receive_permutation(ranking, result, start_pos, end_pos)
                end_pos -= self.step_size
        return ranking

@component
class GroqRankLLMReranker:
    def __init__(self, model_name: str = "mixtral-8x7b-32768", 
                 api_key: Optional[str] = None, window_size: int = 10, 
                 step_size: int = 5, num_repeat: int = 1):
        self.model_name = model_name
        self.api_key = api_key
        self.window_size = window_size
        self.step_size = step_size
        self.num_repeat = num_repeat
        self.ranker = GroqListwiseLlmRanker(
            model_name, api_key, window_size, step_size, num_repeat
        )

    def to_dict(self):
        return default_to_dict(
            self,
            model_name=self.model_name,
            window_size=self.window_size,
            step_size=self.step_size,
            num_repeat=self.num_repeat
        )

    @component.output_types(documents=List[Document])
    def run(self, query: str, documents: List[Document]):
        reranked = self.ranker.rerank(query, documents)
        # Preserve original metadata while updating scores
        return {"documents": [
            Document(
                content=doc.content,
                meta=doc.meta,
                score=-i  # New ranking score
            ) for i, doc in enumerate(reranked)
        ]}