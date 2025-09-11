# paper_agent/parsers/pdf_parser.py

from pypdf import PdfReader
import os
from typing import List, Dict, Any, Optional

from utils.logger import logger


class PDFParser:
    """
    A class for parsing PDF files, extracting text content, and metadata.
    Uses the pypdf library.
    """

    def __init__(self):
        logger.info("PDFParser initialized.")

    def _get_pdf_reader(self, pdf_path: str) -> Optional[PdfReader]:
        """
        Helper to get a PdfReader object for a given PDF path.
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return None
        try:
            reader = PdfReader(pdf_path)
            return reader
        except Exception as e:
            logger.error(f"Error opening or reading PDF '{pdf_path}': {e}")
            return None

    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """
        Extracts all text content from a PDF file.
        """
        reader = self._get_pdf_reader(pdf_path)
        if not reader:
            return None

        full_text = []
        try:
            for page in reader.pages:
                full_text.append(page.extract_text())
            return "\n".join(full_text)
        except Exception as e:
            logger.error(f"Error extracting text from PDF '{pdf_path}': {e}")
            return None

    def extract_metadata_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extracts basic metadata (like title, author) from a PDF file.
        Returns a dictionary with metadata.
        """
        reader = self._get_pdf_reader(pdf_path)
        if not reader:
            return {}

        metadata = reader.metadata
        return {
            "title": metadata.get("/Title"),
            "author": metadata.get("/Author"),
            "creator": metadata.get("/Creator"),
            "producer": metadata.get("/Producer"),
            "creation_date": metadata.get("/CreationDate"),
            "mod_date": metadata.get("/ModDate"),
            "keywords": metadata.get("/Keywords"),
            # Add more fields as needed
        }

    def extract_sections_from_pdf(self, pdf_path: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Extracts text from a PDF and chunks it into manageable sections.
        This is a simple chunking mechanism. More advanced methods (e.g., based on headings,
        semantic similarity) might be needed for better RAG performance.

        Returns a list of dictionaries, each with 'content', 'page_number', 'section_title' (optional).
        """
        reader = self._get_pdf_reader(pdf_path)
        if not reader:
            return []

        sections = []
        current_page = 0
        for page_num, page in enumerate(reader.pages):
            current_page = page_num + 1  # Page numbers are 1-indexed
            text = page.extract_text()
            if not text:
                continue

            # Simple chunking by character count
            start_idx = 0
            while start_idx < len(text):
                end_idx = start_idx + chunk_size
                chunk = text[start_idx:end_idx]

                # Attempt to find a natural break (e.g., end of sentence)
                # This is a very basic attempt; real-world parsing might need NLP
                if end_idx < len(text):
                    last_period = chunk.rfind(".")
                    last_newline = chunk.rfind("\n")
                    split_point = max(last_period, last_newline)

                    if split_point > chunk_size * 0.75:  # If a good split point is near the end
                        chunk = text[start_idx : start_idx + split_point + 1]
                        start_idx += split_point + 1
                    else:
                        start_idx += chunk_size - overlap  # Move back by overlap
                else:
                    start_idx += chunk_size  # Process remaining chunk

                if chunk.strip():  # Add only non-empty chunks
                    sections.append(
                        {
                            "content": chunk.strip(),
                            "page_number": current_page,
                            "section_title": f"Page {current_page} Chunk {len(sections)+1}",  # Placeholder
                        }
                    )

        # TODO: Implement more sophisticated section extraction (e.g., based on headings, TOC)
        # This currently just chunks pages. A real paper parser would use layout analysis
        # to identify Introduction, Methodology, etc. This is a complex task.

        return sections


# For testing purposes
if __name__ == "__main__":
    from utils.config import config
    from utils.logger import logger
    import sys

    # Ensure data directory exists for test PDF
    test_pdf_dir = config.get("PAPERS_DIR")
    os.makedirs(test_pdf_dir, exist_ok=True)

    # Create a dummy PDF for testing if one doesn't exist
    test_pdf_path = os.path.join(test_pdf_dir, "test_paper.pdf")
    if not os.path.exists(test_pdf_path):
        logger.warning(f"No test_paper.pdf found at {test_pdf_path}. Please place a dummy PDF there for testing.")
        logger.warning("You can create a simple PDF with some text using a word processor and save it.")
        sys.exit("Please provide a test PDF to run the parser test.")

    parser = PDFParser()

    print(f"\n--- Testing PDF Text Extraction: {test_pdf_path} ---")
    full_text = parser.extract_text_from_pdf(test_pdf_path)
    if full_text:
        print(f"Extracted {len(full_text)} characters.")
        print("\n--- First 500 chars of full text ---")
        print(full_text[:500])
        print("...")
    else:
        print("Failed to extract full text.")

    print("\n--- Testing PDF Metadata Extraction ---")
    metadata = parser.extract_metadata_from_pdf(test_pdf_path)
    if metadata:
        for key, value in metadata.items():
            print(f"{key}: {value}")
    else:
        print("Failed to extract metadata.")

    print("\n--- Testing PDF Section Extraction (Simple Chunking) ---")
    sections = parser.extract_sections_from_pdf(test_pdf_path, chunk_size=500, overlap=100)
    if sections:
        print(f"Extracted {len(sections)} sections.")
        for i, section in enumerate(sections[:3]):  # Print first 3 sections
            print(f"\nSection {i+1} (Page {section['page_number']}, Title: {section['section_title']}):")
            print(section["content"][:300] + "...")  # Print first 300 chars
    else:
        print("Failed to extract sections or no sections found.")
