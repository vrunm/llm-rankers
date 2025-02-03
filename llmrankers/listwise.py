from haystack import Document, component, default_from_dict, default_to_dict
from typing import List, Optional
import groq
import copy
import tiktoken

@component
class ListwiseRanker:
    def __init__(self, 
                 model_name: str = "mixtral-8x7b-32768",
                 api_key: Optional[str] = None,
                 window_size: int = 10,
                 step_size: int = 5,
                 num_repeat: int = 1):
        """
        Haystack component for listwise LLM ranking using Groq
        """
        self.model_name = model_name
        self.api_key = api_key
        self.window_size = window_size
        self.step_size = step_size
        self.num_repeat = num_repeat
        self.client = groq.Client(api_key=api_key) if api_key else None

    def to_dict(self):
        return default_to_dict(
            self,
            model_name=self.model_name,
            window_size=self.window_size,
            step_size=self.step_size,
            num_repeat=self.num_repeat
        )

    @classmethod
    def from_dict(cls, data):
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, query: str, documents: List[Document]):
        reranked = self._rerank(query, documents)
        return {"documents": [
            Document(
                content=doc.content,
                meta=doc.meta,
                score=-i
            ) for i, doc in enumerate(reranked)
        ]}

    def _create_messages(self, query: str, docs: List[Document]):
        """Build chat messages for ranking instruction"""
        messages = [
            {'role': 'system', 'content': "You are RankGPT, a relevance ranking assistant."},
            {'role': 'user', 'content': f"Rank {len(docs)} passages for query: {query}."},
            {'role': 'assistant', 'content': 'Ready for passages.'}
        ]
        
        max_length = 300
        for i, doc in enumerate(docs):
            content = ' '.join(doc.content.replace('Title: Content: ', '').strip().split()[:max_length])
            messages.extend([
                {'role': 'user', content: f"[{i+1}] {content}"},
                {'role': 'assistant', 'content': f'Received passage [{i+1}]'}
            ])
            
        messages.append({'role': 'user', 'content': 
            f"Search Query: {query}. \nRank the {len(docs)} passages based on relevance. "
            "List in descending order using identifiers like [1] > [2]. Only provide ranking."})
        return messages

    def _rerank(self, query: str, ranking: List[Document]) -> List[Document]:
        """Sliding window reranking implementation"""
        for _ in range(self.num_repeat):
            ranking = copy.deepcopy(ranking)
            end_pos = len(ranking)
            while (start_pos := end_pos - self.window_size) >= 0:
                window_docs = ranking[start_pos:end_pos]
                result = self._compare(query, window_docs)
                ranking = self._process_ranking(ranking, result, start_pos, end_pos)
                end_pos -= self.step_size
        return ranking

    def _compare(self, query: str, docs: List[Document]):
        """Execute LLM comparison via Groq API"""
        messages = self._create_messages(query, docs)
        completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            temperature=0.0
        )
        return self._clean_response(completion.choices[0].message.content)

    def _clean_response(self, response: str):
        """Normalize LLM output for ranking processing"""
        cleaned = ''.join(c if c.isdigit() else ' ' for c in response).strip()
        return sorted(set(cleaned.split()), key=cleaned.split().index)

    def _process_ranking(self, ranking, permutation, start: int, end: int):
        """Apply permutation results to document ordering"""
        try:
            indices = [int(x)-1 for x in permutation]
            window = copy.deepcopy(ranking[start:end])
            return ranking[:start] + [window[i] for i in indices] + ranking[end:]
        except Exception as e:
            print(f"Ranking error: {str(e)}")
            return ranking