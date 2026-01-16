"""
Hybrid Retrieval Module
Combines BM25 (sparse) and vector (dense) retrieval for better results.
"""

import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings
from openai import OpenAI


class HybridRetriever:
    """
    Hybrid retriever combining BM25 and vector search.
    """

    def __init__(
        self,
        collection_name: str = "document_chunks",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "text-embedding-3-small"
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model

        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # BM25 components (will be initialized when documents are added)
        self.bm25 = None
        self.chunks = []
        self.tokenized_corpus = []

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings from OpenAI API.
        """
        # Process in batches to avoid rate limits
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for BM25.
        """
        # Lowercase and split by non-alphanumeric characters
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def index_chunks(self, chunks: List[Dict[str, Any]], force_reindex: bool = False) -> None:
        """
        Index chunks for both vector and BM25 retrieval.
        """
        self.chunks = chunks

        # Check if collection exists
        existing_collections = [c.name for c in self.chroma_client.list_collections()]

        if self.collection_name in existing_collections and not force_reindex:
            # Load existing collection
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            print(f"Loaded existing collection with {self.collection.count()} documents")

            # Rebuild BM25 from stored chunks
            self._rebuild_bm25_from_collection()
            return

        # Delete if exists and force_reindex
        if self.collection_name in existing_collections:
            self.chroma_client.delete_collection(name=self.collection_name)

        # Create new collection
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        print(f"Indexing {len(chunks)} chunks...")

        # Prepare data for ChromaDB
        texts = [chunk['text'] for chunk in chunks]
        ids = [str(chunk['chunk_id']) for chunk in chunks]
        metadatas = [
            {
                'page_num': chunk['page_num'],
                'chunk_id': chunk['chunk_id'],
                'citation': chunk['citation'],
                'token_count': chunk['token_count']
            }
            for chunk in chunks
        ]

        # Get embeddings
        print("Generating embeddings...")
        embeddings = self.get_embeddings(texts)

        # Add to ChromaDB in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )

        print(f"Indexed {len(chunks)} chunks in ChromaDB")

        # Build BM25 index
        self._build_bm25_index(texts)

    def _build_bm25_index(self, texts: List[str]) -> None:
        """
        Build BM25 index from texts.
        """
        self.tokenized_corpus = [self.tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print("Built BM25 index")

    def _rebuild_bm25_from_collection(self) -> None:
        """
        Rebuild BM25 index from ChromaDB collection.
        """
        # Get all documents from collection
        results = self.collection.get(include=['documents', 'metadatas'])

        if results['documents']:
            # Reconstruct chunks list
            self.chunks = []
            for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas'])):
                self.chunks.append({
                    'chunk_id': meta['chunk_id'],
                    'page_num': meta['page_num'],
                    'text': doc,
                    'citation': meta['citation'],
                    'token_count': meta.get('token_count', 0)
                })

            # Sort by chunk_id
            self.chunks.sort(key=lambda x: x['chunk_id'])

            # Build BM25
            texts = [chunk['text'] for chunk in self.chunks]
            self._build_bm25_index(texts)

    def search_vector(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Vector similarity search.
        """
        if not self.collection:
            return []

        # Get query embedding
        query_embedding = self.get_embeddings([query])[0]

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )

        # Process results
        retrieved = []
        if results['documents'] and results['documents'][0]:
            for doc, meta, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                # Convert distance to similarity score (cosine distance to similarity)
                similarity = 1 - distance
                retrieved.append((
                    {
                        'text': doc,
                        'page_num': meta['page_num'],
                        'chunk_id': meta['chunk_id'],
                        'citation': meta['citation']
                    },
                    similarity
                ))

        return retrieved

    def search_bm25(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        BM25 sparse retrieval.
        """
        if not self.bm25 or not self.chunks:
            return []

        # Tokenize query
        query_tokens = self.tokenize(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Build results
        retrieved = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include if there's some match
                chunk = self.chunks[idx]
                retrieved.append((
                    {
                        'text': chunk['text'],
                        'page_num': chunk['page_num'],
                        'chunk_id': chunk['chunk_id'],
                        'citation': chunk['citation']
                    },
                    float(scores[idx])
                ))

        return retrieved

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector and BM25 results.
        Uses Reciprocal Rank Fusion (RRF) for combining rankings.
        """
        # Get results from both methods (fetch more for fusion)
        vector_results = self.search_vector(query, top_k=top_k * 2)
        bm25_results = self.search_bm25(query, top_k=top_k * 2)

        # Combine using RRF
        k = 60  # RRF constant

        # Calculate RRF scores
        rrf_scores = {}

        # Process vector results
        for rank, (chunk, score) in enumerate(vector_results):
            chunk_key = chunk['chunk_id']
            if chunk_key not in rrf_scores:
                rrf_scores[chunk_key] = {
                    'chunk': chunk,
                    'vector_score': score,
                    'bm25_score': 0,
                    'rrf_score': 0
                }
            rrf_scores[chunk_key]['rrf_score'] += vector_weight / (k + rank + 1)
            rrf_scores[chunk_key]['vector_score'] = score

        # Process BM25 results
        for rank, (chunk, score) in enumerate(bm25_results):
            chunk_key = chunk['chunk_id']
            if chunk_key not in rrf_scores:
                rrf_scores[chunk_key] = {
                    'chunk': chunk,
                    'vector_score': 0,
                    'bm25_score': score,
                    'rrf_score': 0
                }
            rrf_scores[chunk_key]['rrf_score'] += bm25_weight / (k + rank + 1)
            rrf_scores[chunk_key]['bm25_score'] = score

        # Sort by RRF score
        sorted_results = sorted(
            rrf_scores.values(),
            key=lambda x: x['rrf_score'],
            reverse=True
        )[:top_k]

        # Format results
        final_results = []
        for item in sorted_results:
            final_results.append({
                'text': item['chunk']['text'],
                'page_num': item['chunk']['page_num'],
                'chunk_id': item['chunk']['chunk_id'],
                'citation': item['chunk']['citation'],
                'vector_score': round(item['vector_score'], 4),
                'bm25_score': round(item['bm25_score'], 4),
                'combined_score': round(item['rrf_score'], 4)
            })

        return final_results

    def clear_index(self) -> None:
        """
        Clear the index and start fresh.
        """
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
        except:
            pass
        self.bm25 = None
        self.chunks = []
        self.tokenized_corpus = []


if __name__ == "__main__":
    # Test retrieval
    retriever = HybridRetriever()

    test_chunks = [
        {'chunk_id': 0, 'page_num': 1, 'text': 'Revenue increased by 20% year over year', 'citation': '[p1:c0]', 'token_count': 8},
        {'chunk_id': 1, 'page_num': 2, 'text': 'EBITDA margin improved to 15%', 'citation': '[p2:c1]', 'token_count': 6},
        {'chunk_id': 2, 'page_num': 3, 'text': 'Airport business saw passenger growth', 'citation': '[p3:c2]', 'token_count': 6},
    ]

    retriever.index_chunks(test_chunks, force_reindex=True)
    results = retriever.search_hybrid("What was the revenue growth?", top_k=2)

    for r in results:
        print(f"{r['citation']}: {r['text'][:100]}...")
        print(f"  Scores - Vector: {r['vector_score']}, BM25: {r['bm25_score']}, Combined: {r['combined_score']}")
