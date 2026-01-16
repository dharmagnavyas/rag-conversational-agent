#!/usr/bin/env python3
"""
Download the sample PDF for testing.
"""

import urllib.request
import os

SAMPLE_PDF_URL = "https://www.adanienterprises.com/-/media/Project/Enterprises/Investors/Investor-Downloads/Results-Presentations/AEL_Earnings_Presentation_Q2-FY26.pdf"
OUTPUT_PATH = "./doc.pdf"


def download_pdf():
    """Download the sample PDF."""
    if os.path.exists(OUTPUT_PATH):
        print(f"File already exists: {OUTPUT_PATH}")
        overwrite = input("Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Skipped download.")
            return

    print(f"Downloading from {SAMPLE_PDF_URL}...")
    print("This may take a moment...")

    try:
        urllib.request.urlretrieve(SAMPLE_PDF_URL, OUTPUT_PATH)
        file_size = os.path.getsize(OUTPUT_PATH)
        print(f"Downloaded successfully: {OUTPUT_PATH} ({file_size / 1024:.1f} KB)")
    except Exception as e:
        print(f"Error downloading: {e}")
        print("\nPlease download manually from:")
        print(SAMPLE_PDF_URL)


if __name__ == "__main__":
    download_pdf()
