# paper_agent/management/paper_manager.py

import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from database.db_manager import DBManager
from parsers.pdf_parser import PDFParser
from embeddings.embedding_model import EmbeddingModel
from rag.retriever import Retriever
from rag.generator import Generator
from llm.llm_interface import LLMInterface  # Needed for summary generation
from utils.logger import logger
from utils.config import config  # For chunking parameters


class PaperManager:
    """
    Manages the lifecycle of papers in the system, orchestrating interactions
    between the database, PDF parser, embedding model, and RAG components.
    """

    def __init__(
        self,
        db_manager: DBManager,
        pdf_parser: PDFParser,
        embedding_model: EmbeddingModel,
        retriever: Retriever,
        generator: Generator,
        llm_interface: LLMInterface,
    ):  # Added llm_interface directly for summary
        self.db_manager = db_manager
        self.pdf_parser = pdf_parser
        self.embedding_model = embedding_model
        self.retriever = retriever
        self.generator = generator
        self.llm = llm_interface  # Store LLM interface for direct use (e.g., summary)

        self.chunk_size = config.get("PDF_CHUNK_SIZE", 1000)
        self.chunk_overlap = config.get("PDF_CHUNK_OVERLAP", 200)

        logger.info("PaperManager initialized.")

    def add_paper_from_file(self, file_path: str) -> Optional[int]:
        """
        Adds a new paper to the system by parsing its PDF, extracting metadata,
        chunking its content, generating embeddings, and storing everything in the database.

        Args:
            file_path (str): The absolute path to the PDF file.

        Returns:
            Optional[int]: The ID of the newly added paper, or None if the operation failed.
        """
        if not os.path.exists(file_path):
            logger.error(f"Attempted to add non-existent file: {file_path}")
            return None

        # 1. Check if paper already exists (by file_path)
        existing_paper = self.db_manager.get_paper_by_filepath(file_path)
        if existing_paper:
            logger.warning(f"Paper from '{file_path}' already exists in DB (ID: {existing_paper['id']}). Skipping.")
            return existing_paper["id"]

        logger.info(f"Adding new paper from: {file_path}")

        # 2. Extract metadata using LLM
        llm_extracted_metadata = self.pdf_parser.extract_metadata_with_llm(file_path)
        if not llm_extracted_metadata:
            logger.error(f"Failed to extract metadata for {file_path} using LLM. Cannot add paper.")
            return None

        # Use extracted metadata; fallback to generic values if LLM fails for a field
        paper_title = llm_extracted_metadata.get("title", os.path.basename(file_path).replace(".pdf", "")).strip()
        paper_authors = llm_extracted_metadata.get("authors", "Unknown Authors").strip()
        paper_abstract = llm_extracted_metadata.get("abstract", "No abstract extracted.").strip()
        # publication_year, doi, url are harder for LLM to consistently extract without more context/prompt engineering
        # For now, we might leave them as None or try to extract from pypdf metadata first
        # We'll use a placeholder for year if LLM doesn't provide it
        # For simplicity, we'll keep the year as a placeholder for now
        paper_year = int(llm_extracted_metadata.get("publication_year", 0))  # Assuming LLM *could* return this
        if paper_year == 0:  # Fallback if LLM didn't find it
            metadata_pypdf = self.pdf_parser.extract_metadata_from_pdf(file_path)
            if (
                "/CreationDate" in metadata_pypdf
                and metadata_pypdf["/CreationDate"]
                and len(metadata_pypdf["/CreationDate"]) >= 5
            ):
                try:
                    paper_year = int(metadata_pypdf["/CreationDate"][2:6])  # e.g., 'D:20230101...'
                except ValueError:
                    pass
        if paper_year == 0:
            paper_year = datetime.now().year  # Default to current year if all else fails

        # 3. Add paper entry to DB
        paper_id = self.db_manager.add_paper(
            title=paper_title,
            authors=paper_authors,
            publication_year=paper_year,
            abstract=paper_abstract,
            file_path=file_path,
        )
        if not paper_id:
            logger.error(f"Failed to add paper '{paper_title}' to database.")
            return None

        logger.info(f"Paper '{paper_title}' (ID: {paper_id}) added to DB metadata.")

        # 4. Extract sections and generate embeddings
        sections_data = self.pdf_parser.extract_sections_from_pdf(
            file_path, chunk_size=self.chunk_size, overlap=self.chunk_overlap
        )
        if not sections_data:
            logger.warning(f"No sections extracted for paper '{paper_title}'. Skipping embedding and section storage.")
            return paper_id  # Paper added, but no sections. Still a success.

        section_contents = [sec["content"] for sec in sections_data]
        embeddings = self.embedding_model.get_embedding(section_contents)

        if embeddings is None or embeddings.shape[0] != len(sections_data):
            logger.error(
                f"Failed to generate embeddings for all sections of '{paper_title}'. Storing sections without embeddings."
            )
            # Store sections without embeddings if embedding generation fails
            for section in sections_data:
                self.db_manager.add_section(paper_id=paper_id, **section, embedding=None)
        else:
            for i, section in enumerate(sections_data):
                self.db_manager.add_section(
                    paper_id=paper_id,
                    section_title=section["section_title"],
                    content=section["content"],
                    page_number=section["page_number"],
                    embedding=embeddings[i],
                )
            logger.info(f"Stored {len(sections_data)} sections with embeddings for paper '{paper_title}'.")

        return paper_id

    def query_papers_rag(self, query: str, top_k_sections: int = 5) -> Dict[str, Any]:
        """
        Performs a RAG query across all ingested papers.

        Args:
            query (str): The user's question.
            top_k_sections (int): Number of top relevant sections to retrieve.

        Returns:
            Dict[str, Any]: A dictionary containing the LLM-generated answer and the
                            retrieved sections used as context.
        """
        logger.info(f"Processing RAG query: '{query}'")
        retrieved_sections = self.retriever.retrieve_relevant_sections(query, top_k=top_k_sections)

        answer = self.generator.generate_answer(query, retrieved_sections)

        return {"query": query, "answer": answer, "retrieved_sections": retrieved_sections}

    def summarize_paper(self, paper_id: int) -> Optional[str]:
        """
        Generates a comprehensive summary of a specific paper using the LLM.
        The summary is then stored in the database.

        Args:
            paper_id (int): The ID of the paper to summarize.

        Returns:
            Optional[str]: The generated summary text, or None if summarization failed.
        """
        paper = self.db_manager.get_paper(paper_id)
        if not paper:
            logger.error(f"Paper with ID {paper_id} not found for summarization.")
            return None

        if paper["is_summarized"] and paper["summary_text"]:
            logger.info(f"Paper '{paper['title']}' (ID: {paper_id}) already summarized.")
            return paper["summary_text"]

        logger.info(f"Generating summary for paper '{paper['title']}' (ID: {paper_id}). This may take a while.")

        # Option 1: Use the stored abstract (quickest, but might not be comprehensive enough)
        # prompt_context = paper['abstract']

        # Option 2: Combine all section contents (most comprehensive, but can be very long for LLM)
        sections = self.db_manager.get_sections_for_paper(paper_id)
        if not sections:
            logger.warning(f"No sections found for paper ID {paper_id}. Cannot summarize comprehensively.")
            # Fallback to abstract if no sections
            prompt_context = paper["abstract"] if paper["abstract"] else "No content available."
        else:
            # Join all section contents. Need to be mindful of LLM context window.
            # For very long papers, a more advanced summarization (e.g., hierarchical) might be needed.
            prompt_context = "\n\n".join([sec["content"] for sec in sections])

        # Limit context size to LLM's capacity (e.g., 16k tokens)
        MAX_CONTEXT_CHARS = config.get("LLM_MAX_CONTEXT_CHARS_FOR_SUMMARY", 10000)  # Roughly 2k-3k tokens
        if len(prompt_context) > MAX_CONTEXT_CHARS:
            logger.warning(
                f"Context for summarization too long ({len(prompt_context)} chars). Truncating to {MAX_CONTEXT_CHARS} chars."
            )
            prompt_context = prompt_context[:MAX_CONTEXT_CHARS] + "\n\n[...truncated for brevity...]"

        summary_prompt = f"""
        Please provide a comprehensive and concise summary of the following academic paper.
        Focus on the main objectives, methodology, key findings, and conclusions.
        Ensure the summary is easy to understand for a non-expert, yet captures the essence of the research.

        Paper Content:
        ---
        {prompt_context}
        ---

        Comprehensive Summary:
        """
        summary = self.llm.generate_text(
            summary_prompt, max_tokens=config.get("LLM_MAX_TOKENS_FOR_SUMMARY", 750), temperature=0.5
        )

        if summary:
            self.db_manager.update_paper_summary(paper_id, summary)
            logger.info(f"Summary generated and stored for paper '{paper['title']}' (ID: {paper_id}).")
            return summary
        else:
            logger.error(f"Failed to generate summary for paper '{paper['title']}' (ID: {paper_id}).")
            return None

    def list_all_papers(self) -> List[Dict[str, Any]]:
        """
        Retrieves a list of all papers in the database.
        """
        return self.db_manager.get_all_papers()

    def get_paper_details(self, paper_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves all details for a specific paper, including its tags and summary.
        """
        paper = self.db_manager.get_paper(paper_id)
        if paper:
            paper["tags"] = self.db_manager.get_tags_for_paper(paper_id)
            paper["references"] = self.db_manager.get_paper_references_for_paper(paper_id)
            # Sections are usually too verbose for 'details', but could be added if needed
        return paper

    def add_tag_to_paper(self, paper_id: int, tag_name: str) -> bool:
        """
        Adds a tag to a paper. Creates the tag if it doesn't exist.
        """
        tag_id = self.db_manager.add_tag(tag_name)
        if tag_id:
            return self.db_manager.add_paper_tag(paper_id, tag_id)
        return False

    def remove_tag_from_paper(self, paper_id: int, tag_name: str) -> bool:
        """
        Removes a tag from a paper.
        """
        tag = self.db_manager.get_tag_by_name(tag_name)
        if tag:
            return self.db_manager.remove_paper_tag(paper_id, tag["id"])
        return False

    def delete_paper(self, paper_id: int) -> bool:
        """
        Deletes a paper and all its associated data from the system.
        """
        paper = self.db_manager.get_paper(paper_id)
        if not paper:
            logger.warning(f"Attempted to delete non-existent paper with ID: {paper_id}.")
            return False
        logger.info(f"Deleting paper '{paper['title']}' (ID: {paper_id}) and all associated data.")
        return self.db_manager.delete_paper(paper_id)
