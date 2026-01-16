"""
Conversational RAG Agent
Handles multi-turn conversations with grounded answers and citations.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from retriever import HybridRetriever


class ConversationalAgent:
    """
    RAG-based conversational agent with citation support.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        model: str = "gpt-4o",
        temperature: float = 0.2,
        max_history_turns: int = 10
    ):
        self.retriever = retriever
        self.model = model
        self.temperature = temperature
        self.max_history_turns = max_history_turns

        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.conversation_history: List[Dict[str, str]] = []

        # System prompts
        self.reformulation_prompt = """You are a query reformulator. Given the conversation history and the latest user question,
reformulate the question to be self-contained (capturing all necessary context from the conversation).
If the question is already self-contained, return it as-is.
Only output the reformulated question, nothing else."""

        self.answer_prompt = """You are a helpful assistant that answers questions based ONLY on the provided document context.

CRITICAL RULES:
1. ONLY use information from the provided context chunks to answer
2. If the answer is NOT in the context, respond EXACTLY with: "Not found in the document."
3. NEVER make up or hallucinate information
4. ALWAYS include citations in format [pX] or [pX:cY] where X is page number and Y is chunk ID
5. For numeric questions, only provide numbers that appear verbatim in the context
6. Cite the specific chunk(s) that support each part of your answer

FORMAT YOUR RESPONSE AS:
**Answer:** [Your concise answer with inline citations]

**Evidence:**
- [Citation]: Relevant quote or paraphrase from that chunk

If information is not found, respond with:
**Answer:** Not found in the document."""

    def reformulate_query(self, query: str) -> str:
        """
        Reformulate query using conversation history for context.
        """
        if not self.conversation_history:
            return query

        # Build conversation context
        history_text = ""
        for turn in self.conversation_history[-4:]:  # Last 4 turns
            history_text += f"User: {turn['user']}\nAssistant: {turn['assistant'][:200]}...\n\n"

        messages = [
            {"role": "system", "content": self.reformulation_prompt},
            {"role": "user", "content": f"""Conversation history:
{history_text}

Latest question: {query}

Reformulated question:"""}
        ]

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Use smaller model for reformulation
            messages=messages,
            temperature=0,
            max_tokens=200
        )

        return response.choices[0].message.content.strip()

    def build_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Build context string from retrieved chunks.
        """
        if not retrieved_chunks:
            return "No relevant context found in the document."

        context_parts = []
        for chunk in retrieved_chunks:
            citation = chunk['citation']
            text = chunk['text']
            context_parts.append(f"[{citation}]\n{text}")

        return "\n\n---\n\n".join(context_parts)

    def generate_answer(
        self,
        query: str,
        context: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Generate grounded answer with citations.
        """
        # Build the prompt
        messages = [
            {"role": "system", "content": self.answer_prompt},
            {"role": "user", "content": f"""CONTEXT FROM DOCUMENT:
{context}

USER QUESTION: {query}

Remember:
- Only answer using the context above
- Include citations [pX:cY] for page X, chunk Y
- If not found in context, say "Not found in the document."
"""}
        ]

        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=1000
        )

        answer = response.choices[0].message.content.strip()

        # Validate answer - check for hallucination signals
        answer = self._validate_answer(answer, retrieved_chunks)

        return answer

    def _validate_answer(
        self,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Validate answer to prevent hallucination.
        """
        # If no chunks were retrieved, force "not found"
        if not retrieved_chunks:
            return "**Answer:** Not found in the document."

        # Check if the answer seems to contain information
        lower_answer = answer.lower()

        # If answer already says not found, return it
        if "not found in the document" in lower_answer:
            return answer

        # Check if answer has citations
        has_citations = "[p" in answer or "[c" in answer

        if not has_citations:
            # Try to add relevant citations based on what chunks were used
            available_citations = [c['citation'] for c in retrieved_chunks[:3]]
            if available_citations:
                answer += f"\n\n*Sources: {', '.join(available_citations)}*"

        return answer

    def ask(
        self,
        query: str,
        top_k: int = 5,
        show_debug: bool = True
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process a question and return answer with retrieved chunks.
        """
        # Step 1: Reformulate query if there's conversation history
        reformulated_query = self.reformulate_query(query)

        if show_debug and reformulated_query != query:
            print(f"\n[DEBUG] Reformulated query: {reformulated_query}")

        # Step 2: Retrieve relevant chunks using hybrid search
        retrieved_chunks = self.retriever.search_hybrid(
            reformulated_query,
            top_k=top_k
        )

        # Step 3: Build context from retrieved chunks
        context = self.build_context(retrieved_chunks)

        # Step 4: Generate answer
        answer = self.generate_answer(reformulated_query, context, retrieved_chunks)

        # Step 5: Update conversation history
        self.conversation_history.append({
            'user': query,
            'assistant': answer,
            'retrieved_chunks': retrieved_chunks
        })

        # Trim history if too long
        if len(self.conversation_history) > self.max_history_turns:
            self.conversation_history = self.conversation_history[-self.max_history_turns:]

        return answer, retrieved_chunks

    def display_retrieval_debug(self, retrieved_chunks: List[Dict[str, Any]]) -> None:
        """
        Display debug information about retrieved chunks.
        """
        print("\n" + "=" * 60)
        print("RETRIEVAL DEBUG INFO")
        print("=" * 60)

        if not retrieved_chunks:
            print("No chunks retrieved.")
            return

        for i, chunk in enumerate(retrieved_chunks, 1):
            print(f"\n--- Chunk {i} {chunk['citation']} ---")
            print(f"Page: {chunk['page_num']}, Chunk ID: {chunk['chunk_id']}")
            print(f"Scores - Vector: {chunk.get('vector_score', 'N/A')}, "
                  f"BM25: {chunk.get('bm25_score', 'N/A')}, "
                  f"Combined: {chunk.get('combined_score', 'N/A')}")
            print(f"Text snippet: {chunk['text'][:300]}...")

        print("\n" + "=" * 60)

    def clear_history(self) -> None:
        """
        Clear conversation history.
        """
        self.conversation_history = []


def format_answer_for_display(answer: str) -> str:
    """
    Format the answer for terminal display.
    """
    # Add some visual separation
    lines = answer.split('\n')
    formatted = []

    for line in lines:
        if line.startswith('**Answer:**'):
            formatted.append('\n' + line)
        elif line.startswith('**Evidence:**'):
            formatted.append('\n' + line)
        else:
            formatted.append(line)

    return '\n'.join(formatted)


if __name__ == "__main__":
    # Quick test
    from retriever import HybridRetriever

    retriever = HybridRetriever()
    agent = ConversationalAgent(retriever)

    print("Agent initialized. Use main.py for full functionality.")
