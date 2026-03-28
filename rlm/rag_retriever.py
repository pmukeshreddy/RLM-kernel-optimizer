import math
import glob
import os
import re
from pathlib import Path
from collections import Counter

class BM25Retriever:
    """A zero-dependency BM25 text retriever for local RAG implementations."""
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = []   # List of dicts: {'path': str, 'content': str, 'title': str}
        self.doc_tokens = []  # Tokenized documents
        self.df = Counter()   # Document frequency of terms
        self.idf = {}         # Inverse document frequency
        self.doc_len = []     # Length of each document
        self.avgdl = 0        # Average document length
        self.N = 0            # Total documents
        
    def _tokenize(self, text):
        return [word.lower() for word in re.findall(r'\b\w+\b', text)]
        
    def add_documents_from_dir(self, directory_path: str):
        path = Path(directory_path)
        if not path.exists() or not path.is_dir():
            return
            
        for file_path in path.glob("*.md"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            title = content.split('\n')[0].replace('# ', '').strip() if content.startswith('#') else file_path.stem
            
            self.documents.append({
                'path': str(file_path),
                'content': content,
                'title': title
            })
            tokens = self._tokenize(content)
            self.doc_tokens.append(tokens)
            self.doc_len.append(len(tokens))
            
            # Update Document Frequency (for IDF)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.df[token] += 1
                
        self.N = len(self.documents)
        if self.N > 0:
            self.avgdl = sum(self.doc_len) / self.N
            # Precompute IDF
            for word, freq in self.df.items():
                self.idf[word] = math.log(1 + (self.N - freq + 0.5) / (freq + 0.5))
                
    def get_top_k(self, query: str, k: int = 1):
        if self.N == 0:
            return []
            
        query_tokens = self._tokenize(query)
        scores = []
        
        for i, doc_tokens in enumerate(self.doc_tokens):
            score = 0
            doc_freq = Counter(doc_tokens)
            for q_token in query_tokens:
                if q_token not in self.df:
                    continue
                tf = doc_freq[q_token]
                idf = self.idf[q_token]
                
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * (self.doc_len[i] / self.avgdl))
                score += idf * (num / den)
                
            scores.append((score, self.documents[i]))
            
        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # Return only documents with a positive score
        return [doc for score, doc in scores[:k] if score > 0]

def init_knowledge_base(kb_path: str = None) -> BM25Retriever:
    if kb_path is None:
        kb_path = os.path.join(os.path.dirname(__file__), "knowledge_base")
    retriever = BM25Retriever()
    retriever.add_documents_from_dir(kb_path)
    return retriever

if __name__ == "__main__":
    retriever = init_knowledge_base()
    res = retriever.get_top_k("How to bypass L2 cache allocation when doing global writes?")
    for doc in res:
        print(f"Match: {doc['title']}\n{doc['content'][:200]}...")
