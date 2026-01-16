"""
PDF Processing Module
Handles extraction of text from PDFs with page-level metadata.
Uses pdfplumber for better table and layout handling.
"""

import pdfplumber
from typing import List, Dict, Any
import re


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from PDF with page numbers.
    Returns list of dicts with 'page_num', 'text', and 'tables' keys.
    """
    pages_data = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract main text
                text = page.extract_text() or ""

                # Extract tables separately for better handling
                tables = []
                page_tables = page.extract_tables()
                if page_tables:
                    for table in page_tables:
                        if table:
                            # Convert table to text format
                            table_text = convert_table_to_text(table)
                            tables.append(table_text)

                # Combine text and tables
                full_text = text
                if tables:
                    full_text += "\n\n[TABLE DATA]\n" + "\n".join(tables)

                # Clean up the text
                full_text = clean_text(full_text)

                pages_data.append({
                    'page_num': page_num,
                    'text': full_text,
                    'has_tables': len(tables) > 0
                })

    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise

    return pages_data


def convert_table_to_text(table: List[List]) -> str:
    """
    Convert a table (list of rows) to readable text format.
    """
    if not table:
        return ""

    rows = []
    for row in table:
        if row:
            # Clean each cell and join with separator
            cells = [str(cell).strip() if cell else "" for cell in row]
            # Filter out completely empty rows
            if any(cells):
                rows.append(" | ".join(cells))

    return "\n".join(rows)


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing excessive whitespace and artifacts.
    """
    if not text:
        return ""

    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)

    # Replace multiple newlines with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


def get_pdf_metadata(pdf_path: str) -> Dict[str, Any]:
    """
    Extract metadata from PDF.
    """
    metadata = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            metadata['num_pages'] = len(pdf.pages)
            if pdf.metadata:
                metadata.update(pdf.metadata)
    except Exception as e:
        print(f"Error extracting metadata: {e}")

    return metadata


if __name__ == "__main__":
    # Test extraction
    import sys
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        pages = extract_text_from_pdf(pdf_path)
        for page in pages[:3]:  # Show first 3 pages
            print(f"\n--- Page {page['page_num']} ---")
            print(page['text'][:500])
