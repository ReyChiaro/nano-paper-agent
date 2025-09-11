# paper_agent/parsers/pdf_parser.py

from pypdf import PdfReader
import os
from typing import List, Dict, Any, Optional
import json

from utils.logger import logger

# Import LLMInterface
from llm.llm_interface import LLMInterface


class PDFParser:
    """
    A class for parsing PDF files, extracting text content, and metadata.
    Uses the pypdf library and leverages an LLM for enhanced metadata extraction.
    """

    def __init__(self, llm_interface: LLMInterface):
        self.llm = llm_interface
        logger.info("PDFParser initialized with LLM support.")

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

    def extract_text_from_page_range(self, pdf_path: str, start_page: int = 1, end_page: int = 2) -> Optional[str]:
        """
        Extracts text content from a specific range of pages (1-indexed).
        Useful for feeding initial pages to an LLM for metadata extraction.
        """
        reader = self._get_pdf_reader(pdf_path)
        if not reader:
            return None

        extracted_text = []
        try:
            # Adjust to 0-indexed for pypdf
            start_idx = max(0, start_page - 1)
            end_idx = min(len(reader.pages), end_page)

            for i in range(start_idx, end_idx):
                page = reader.pages[i]
                text = page.extract_text()
                if text:
                    extracted_text.append(text)
            return "\n".join(extracted_text)
        except Exception as e:
            logger.error(f"Error extracting text from page range {start_page}-{end_page} of PDF '{pdf_path}': {e}")
            return None

    def extract_metadata_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extracts basic metadata (like title, author) from a PDF file using pypdf's built-in capabilities.
        This serves as a fallback or initial quick check.
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

    def extract_metadata_with_llm(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """
        Extracts structured metadata (title, authors, abstract, abstract summary)
        from the first few pages of a PDF using an LLM.
        """
        # Extract text from the first two pages
        context_text = self.extract_text_from_page_range(pdf_path, start_page=1, end_page=2)
        if not context_text:
            logger.error(f"Could not extract text from first two pages of {pdf_path} for LLM metadata extraction.")
            return None

        prompt = f"""
        You are an expert in academic paper analysis. Your task is to extract key metadata from the provided text, which typically comes from the first few pages of a research paper.
        Identify the paper's title, a comma-separated list of authors, the full abstract, and a very short (1-2 sentences) summary of the abstract.
        Return the information in a JSON object with the following keys:
        - "title": (string) The full title of the paper.
        - "authors": (string) A comma-separated list of all authors.
        - "abstract": (string) The complete abstract of the paper.
        - "abstract_summary": (string) A concise, 1-2 sentence summary of the abstract.

        If any piece of information is not clearly present in the text, use "N/A" for that field.

        Paper Text:
        ---
        {context_text[:4000]} # Limit input to LLM to avoid very long contexts
        ---
        """
        # Using generate_json for structured output
        extracted_data = self.llm.generate_json(prompt)

        if extracted_data:
            # Clean up common LLM artifacts or formatting issues
            for key in ["title", "authors", "abstract", "abstract_summary"]:
                if extracted_data.get(key) and isinstance(extracted_data[key], str):
                    extracted_data[key] = extracted_data[key].strip()

            logger.info(f"LLM successfully extracted metadata for '{extracted_data.get('title', 'N/A')}'")
            return extracted_data
        else:
            logger.error(f"LLM failed to extract metadata for {pdf_path}.")
            return None

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


# For testing purposes (updated to reflect LLM integration)
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

    # Initialize LLMInterface for the parser
    llm_interface_test = LLMInterface()
    parser = PDFParser(llm_interface_test)

    print(f"\n--- Testing PDF Text Extraction: {test_pdf_path} ---")
    full_text = parser.extract_text_from_pdf(test_pdf_path)
    if full_text:
        print(f"Extracted {len(full_text)} characters.")
        print("\n--- First 500 chars of full text ---")
        print(full_text[:500])
        print("...")
    else:
        print("Failed to extract full text.")

    print("\n--- Testing PDF Metadata Extraction (pypdf's built-in) ---")
    metadata_pypdf = parser.extract_metadata_from_pdf(test_pdf_path)
    if metadata_pypdf:
        for key, value in metadata_pypdf.items():
            print(f"pypdf - {key}: {value}")
    else:
        print("Failed to extract metadata using pypdf.")

    print("\n--- Testing PDF Metadata Extraction (LLM-Enhanced) ---")
    metadata_llm = parser.extract_metadata_with_llm(test_pdf_path)
    if metadata_llm:
        for key, value in metadata_llm.items():
            print(f"LLM - {key}: {value}")
    else:
        print("Failed to extract metadata using LLM.")

    print("\n--- Testing PDF Section Extraction (Simple Chunking) ---")
    sections = parser.extract_sections_from_pdf(test_pdf_path, chunk_size=500, overlap=100)
    if sections:
        print(f"Extracted {len(sections)} sections.")
        for i, section in enumerate(sections[:3]):  # Print first 3 sections
            print(f"\nSection {i+1} (Page {section['page_number']}, Title: {section['section_title']}):")
            print(section["content"][:300] + "...")  # Print first 300 chars
    else:
        print("Failed to extract sections or no sections found.")
