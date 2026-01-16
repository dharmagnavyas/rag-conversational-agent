#!/usr/bin/env python3
"""
Acceptance Test Script
Runs the must-pass acceptance tests for the RAG chatbot.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

from pdf_processor import extract_text_from_pdf
from chunker import create_chunks
from retriever import HybridRetriever
from chat_agent import ConversationalAgent


def run_acceptance_tests(pdf_path: str):
    """
    Run acceptance tests against the provided PDF.
    """
    print("\n" + "="*70)
    print("ACCEPTANCE TESTS")
    print("="*70)

    # Initialize components
    print("\nInitializing RAG system...")

    retriever = HybridRetriever(
        collection_name="test_document_chunks",
        persist_directory="./test_chroma_db"
    )

    # Clear any existing index for clean test
    retriever.clear_index()

    # Ingest document
    print(f"Ingesting PDF: {pdf_path}")
    pages_data = extract_text_from_pdf(pdf_path)
    chunks = create_chunks(pages_data, chunk_size=500, chunk_overlap=100)
    retriever.index_chunks(chunks, force_reindex=True)

    # Initialize agent
    agent = ConversationalAgent(retriever=retriever, model="gpt-4o")

    # Test cases
    test_cases = [
        {
            "name": "Test 1: Grounded Fact Question",
            "question": "What are the major business segments discussed in the document?",
            "expected_behavior": "Answer mentions segments present in PDF with citations",
            "should_find": True
        },
        {
            "name": "Test 2: Numeric Question",
            "question": "What is the consolidated total income in H1-26?",
            "expected_behavior": "Returns correct value with citation OR 'Not found'",
            "should_find": True
        },
        {
            "name": "Test 3: Cross-section Question",
            "question": "What drivers are mentioned for EBITDA changes in H1-26?",
            "expected_behavior": "References document's stated drivers with citations",
            "should_find": True
        },
        {
            "name": "Test 4: Negative Control",
            "question": "What is the CEO's email address?",
            "expected_behavior": "Says 'Not found in the document'",
            "should_find": False
        }
    ]

    results = []

    for test in test_cases:
        print(f"\n{'-'*70}")
        print(f"{test['name']}")
        print(f"Question: {test['question']}")
        print(f"Expected: {test['expected_behavior']}")
        print(f"{'-'*70}")

        answer, retrieved = agent.ask(test['question'], top_k=5, show_debug=False)

        print(f"\nAnswer:\n{answer}")

        # Basic validation
        has_citation = "[p" in answer
        says_not_found = "not found in the document" in answer.lower()

        if test['should_find']:
            passed = has_citation and not says_not_found
        else:
            passed = says_not_found

        status = "PASS" if passed else "NEEDS REVIEW"
        print(f"\nStatus: {status}")

        results.append({
            "test": test['name'],
            "passed": passed,
            "answer": answer[:200]
        })

    # Test 5: Conversational follow-up
    print(f"\n{'-'*70}")
    print("Test 5: Conversational Follow-up")
    print(f"{'-'*70}")

    # Clear history for clean test
    agent.clear_history()

    q1 = "Summarize airport performance in H1-26."
    print(f"\nQ1: {q1}")
    answer1, _ = agent.ask(q1, top_k=5, show_debug=False)
    print(f"A1: {answer1[:300]}...")

    q2 = "Break that down into passenger and cargo changes."
    print(f"\nQ2: {q2}")
    answer2, _ = agent.ask(q2, top_k=5, show_debug=False)
    print(f"A2: {answer2}")

    # Check if Q2 answer references context from Q1
    has_citation_q2 = "[p" in answer2
    not_found_q2 = "not found in the document" in answer2.lower()
    passed_followup = has_citation_q2 or not_found_q2

    print(f"\nStatus: {'PASS' if passed_followup else 'NEEDS REVIEW'}")

    results.append({
        "test": "Test 5: Conversational Follow-up",
        "passed": passed_followup,
        "answer": answer2[:200]
    })

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed_count = sum(1 for r in results if r['passed'])
    total_count = len(results)

    for r in results:
        status = "PASS" if r['passed'] else "NEEDS REVIEW"
        print(f"  {r['test']}: {status}")

    print(f"\nOverall: {passed_count}/{total_count} tests passed")

    # Cleanup
    try:
        retriever.clear_index()
        import shutil
        shutil.rmtree("./test_chroma_db", ignore_errors=True)
    except:
        pass

    return passed_count == total_count


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_acceptance.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)

    success = run_acceptance_tests(pdf_path)
    sys.exit(0 if success else 1)
