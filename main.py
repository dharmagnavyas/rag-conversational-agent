#!/usr/bin/env python3
"""
RAG Conversational Agent - Main Entry Point

A conversational agent that answers questions grounded in PDF documents
using Retrieval-Augmented Generation (RAG).

Usage:
    python main.py --pdf <path_to_pdf>
    python main.py --pdf <path_to_pdf> --reindex
    python main.py  # Uses previously indexed document

Features:
    - Hybrid retrieval (BM25 + vector embeddings)
    - Multi-turn conversation with history awareness
    - Grounded answers with citations [pX:cY]
    - Retrieval debug visibility
"""

import os
import sys
import argparse
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from pdf_processor import extract_text_from_pdf, get_pdf_metadata
from chunker import create_chunks
from retriever import HybridRetriever
from chat_agent import ConversationalAgent, format_answer_for_display


class RAGChatbot:
    """
    Main chatbot class that orchestrates PDF ingestion and Q&A.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        model: str = "gpt-4o-mini"
    ):
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = model

        # Initialize retriever
        self.retriever = HybridRetriever(
            collection_name="document_chunks",
            persist_directory=persist_directory
        )

        # Initialize agent
        self.agent = ConversationalAgent(
            retriever=self.retriever,
            model=model
        )

        self.is_indexed = False
        self.document_name = None

    def ingest_pdf(self, pdf_path: str, force_reindex: bool = False) -> bool:
        """
        Ingest and index a PDF document.
        """
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file not found: {pdf_path}")
            return False

        print(f"\n{'='*60}")
        print(f"INGESTING DOCUMENT: {os.path.basename(pdf_path)}")
        print(f"{'='*60}")

        try:
            # Get metadata
            metadata = get_pdf_metadata(pdf_path)
            print(f"Pages: {metadata.get('num_pages', 'Unknown')}")

            # Extract text from PDF
            print("\nExtracting text from PDF...")
            pages_data = extract_text_from_pdf(pdf_path)
            print(f"Extracted text from {len(pages_data)} pages")

            # Create chunks
            print("\nChunking text...")
            chunks = create_chunks(
                pages_data,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            print(f"Created {len(chunks)} chunks")

            # Index chunks
            print("\nIndexing chunks...")
            self.retriever.index_chunks(chunks, force_reindex=force_reindex)

            self.is_indexed = True
            self.document_name = os.path.basename(pdf_path)

            print(f"\nDocument indexed successfully!")
            print(f"{'='*60}\n")

            return True

        except Exception as e:
            print(f"Error during ingestion: {e}")
            import traceback
            traceback.print_exc()
            return False

    def check_existing_index(self) -> bool:
        """
        Check if there's an existing index.
        """
        try:
            collections = self.retriever.chroma_client.list_collections()
            for col in collections:
                if col.name == "document_chunks":
                    count = self.retriever.chroma_client.get_collection("document_chunks").count()
                    if count > 0:
                        self.is_indexed = True
                        # Rebuild BM25
                        self.retriever.collection = self.retriever.chroma_client.get_collection("document_chunks")
                        self.retriever._rebuild_bm25_from_collection()
                        return True
        except:
            pass
        return False

    def chat(self, show_debug: bool = True) -> None:
        """
        Start interactive chat loop.
        """
        if not self.is_indexed:
            if not self.check_existing_index():
                print("\nNo document indexed. Please provide a PDF path.")
                print("Usage: python main.py --pdf <path_to_pdf>")
                return

        print("\n" + "="*60)
        print("RAG CONVERSATIONAL AGENT")
        print("="*60)
        print("\nCommands:")
        print("  /debug    - Toggle retrieval debug info")
        print("  /clear    - Clear conversation history")
        print("  /quit     - Exit the chat")
        print("  /help     - Show this help message")
        print("\n" + "-"*60)

        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() == '/quit' or user_input.lower() == '/exit':
                    print("\nGoodbye!")
                    break

                elif user_input.lower() == '/debug':
                    show_debug = not show_debug
                    status = "ON" if show_debug else "OFF"
                    print(f"\nRetrieval debug mode: {status}")
                    continue

                elif user_input.lower() == '/clear':
                    self.agent.clear_history()
                    print("\nConversation history cleared.")
                    continue

                elif user_input.lower() == '/help':
                    print("\nCommands:")
                    print("  /debug    - Toggle retrieval debug info")
                    print("  /clear    - Clear conversation history")
                    print("  /quit     - Exit the chat")
                    print("  /help     - Show this help message")
                    continue

                # Process question
                print("\nSearching document...")
                answer, retrieved_chunks = self.agent.ask(
                    user_input,
                    top_k=5,
                    show_debug=show_debug
                )

                # Show retrieval debug if enabled
                if show_debug:
                    self.agent.display_retrieval_debug(retrieved_chunks)

                # Display answer
                print("\n" + "-"*60)
                print("ASSISTANT:")
                print(format_answer_for_display(answer))
                print("-"*60)

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type /quit to exit.")
                continue
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()


def download_pdf(url: str, output_path: str) -> bool:
    """
    Download PDF from URL.
    """
    import urllib.request

    print(f"Downloading PDF from {url}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded to {output_path}")
        return True
    except Exception as e:
        print(f"Failed to download: {e}")
        return False


def main():
    """
    Main entry point.
    """
    parser = argparse.ArgumentParser(
        description="RAG Conversational Agent for PDF Q&A",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --pdf document.pdf
    python main.py --pdf document.pdf --reindex
    python main.py --url https://example.com/doc.pdf
    python main.py  # Use existing index
        """
    )

    parser.add_argument(
        '--pdf',
        type=str,
        help='Path to PDF file to ingest'
    )

    parser.add_argument(
        '--url',
        type=str,
        help='URL to download PDF from'
    )

    parser.add_argument(
        '--reindex',
        action='store_true',
        help='Force reindexing even if index exists'
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        default=500,
        help='Chunk size in tokens (default: 500)'
    )

    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=100,
        help='Chunk overlap in tokens (default: 100)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        help='OpenAI model to use (default: gpt-4o)'
    )

    parser.add_argument(
        '--no-debug',
        action='store_true',
        help='Disable retrieval debug output by default'
    )

    args = parser.parse_args()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it: export OPENAI_API_KEY='your-key-here'")
        print("Or create a .env file with OPENAI_API_KEY=your-key-here")
        sys.exit(1)

    # Initialize chatbot
    chatbot = RAGChatbot(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        model=args.model
    )

    # Handle PDF ingestion
    pdf_path = args.pdf

    # Download from URL if provided
    if args.url:
        pdf_path = "./downloaded_document.pdf"
        if not download_pdf(args.url, pdf_path):
            sys.exit(1)

    # Ingest PDF if provided
    if pdf_path:
        if not chatbot.ingest_pdf(pdf_path, force_reindex=args.reindex):
            sys.exit(1)

    # Start chat
    chatbot.chat(show_debug=not args.no_debug)


if __name__ == "__main__":
    main()
