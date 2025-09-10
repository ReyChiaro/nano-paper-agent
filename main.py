# paper_agent/main.py

import os
from utils.config import config
from utils.logger import logger
from database.db_manager import DBManager # Import the new DBManager
import numpy as np # For testing embeddings

def main():
    """
    Main entry point for the Paper Agent application.
    Initializes core components and starts the UI.
    """
    logger.info("Starting Paper Agent application...")

    # --- Configuration and Directory Setup ---
    papers_dir = config.get("PAPERS_DIR")
    db_dir = config.get("DB_DIR")

    os.makedirs(papers_dir, exist_ok=True)
    os.makedirs(db_dir, exist_ok=True)
    logger.info(f"Papers directory: {papers_dir}")
    logger.info(f"Database directory: {db_dir}")
    logger.info(f"Database name: {config.get('DATABASE_NAME')}")

    # --- Initialize DBManager ---
    try:
        db_manager = DBManager()
        logger.info("DBManager initialized successfully.")

        # --- Basic DBManager Test ---
        logger.info("\n--- Testing DBManager operations ---")

        # 1. Add a paper
        paper_id_1 = db_manager.add_paper(
            title="A Novel Approach to Paper Management",
            authors="John Doe, Jane Smith",
            publication_year=2023,
            abstract="This paper introduces a revolutionary system...",
            file_path="/path/to/paper1.pdf",
            doi="10.1234/paper.2023.1"
        )
        if paper_id_1:
            logger.info(f"Test Paper 1 added with ID: {paper_id_1}")
        else:
            logger.error("Failed to add Test Paper 1.")

        paper_id_2 = db_manager.add_paper(
            title="AI for Research Productivity",
            authors="Alice Wonderland",
            publication_year=2024,
            abstract="Exploring how AI can boost research workflow.",
            file_path="/path/to/paper2.pdf",
            url="http://example.com/paper2"
        )
        if paper_id_2:
            logger.info(f"Test Paper 2 added with ID: {paper_id_2}")
        else:
            logger.error("Failed to add Test Paper 2.")

        # Try adding paper with same unique file_path (should fail for file_path)
        paper_id_fail = db_manager.add_paper(
            title="Duplicate Paper Attempt",
            file_path="/path/to/paper1.pdf" # Duplicate file path
        )
        if paper_id_fail is None:
            logger.info("Attempt to add duplicate paper (same file_path) correctly failed.")
        else:
            logger.error(f"Duplicate paper was added with ID: {paper_id_fail}. This is an error.")


        # 2. Get paper by ID
        retrieved_paper = db_manager.get_paper(paper_id_1)
        if retrieved_paper:
            logger.info(f"Retrieved Paper 1: {retrieved_paper['title']} by {retrieved_paper['authors']}")
        else:
            logger.warning(f"Paper with ID {paper_id_1} not found.")

        # 3. Add tags
        tag_nlp_id = db_manager.add_tag("NLP")
        tag_rag_id = db_manager.add_tag("RAG")
        tag_ai_id = db_manager.add_tag("AI")
        logger.info(f"Tags added: NLP ID={tag_nlp_id}, RAG ID={tag_rag_id}, AI ID={tag_ai_id}")

        # 4. Associate tags with papers
        if paper_id_1 and tag_nlp_id: db_manager.add_paper_tag(paper_id_1, tag_nlp_id)
        if paper_id_1 and tag_rag_id: db_manager.add_paper_tag(paper_id_1, tag_rag_id)
        if paper_id_2 and tag_ai_id: db_manager.add_paper_tag(paper_id_2, tag_ai_id)
        if paper_id_2 and tag_rag_id: db_manager.add_paper_tag(paper_id_2, tag_rag_id) # Paper 2 also has RAG

        # 5. Get tags for a paper
        paper1_tags = db_manager.get_tags_for_paper(paper_id_1)
        logger.info(f"Tags for Paper 1 (ID: {paper_id_1}): {[t['name'] for t in paper1_tags]}")
        paper2_tags = db_manager.get_tags_for_paper(paper_id_2)
        logger.info(f"Tags for Paper 2 (ID: {paper_id_2}): {[t['name'] for t in paper2_tags]}")

        # 6. Get papers by tag
        rag_papers = db_manager.get_papers_by_tag("RAG")
        logger.info(f"Papers tagged 'RAG': {[p['title'] for p in rag_papers]}")

        # 7. Add sections with dummy embeddings
        dummy_embedding_1 = np.random.rand(768).astype(np.float32) # Example embedding vector
        dummy_embedding_2 = np.random.rand(768).astype(np.float32)

        if paper_id_1:
            db_manager.add_section(paper_id_1, "Introduction", "This is the intro content.", 1, dummy_embedding_1)
            db_manager.add_section(paper_id_1, "Methodology", "Details of the method.", 3, dummy_embedding_2)

        # 8. Get sections for a paper
        paper1_sections = db_manager.get_sections_for_paper(paper_id_1)
        logger.info(f"Sections for Paper 1 (ID: {paper_id_1}):")
        for sec in paper1_sections:
            logger.info(f"  - {sec['section_title']} (Page: {sec['page_number']}), Embedding shape: {sec['embedding'].shape if sec['embedding'] is not None else 'None'}")

        # 9. Add references
        if paper_id_1:
            db_manager.add_reference(paper_id_1, "Related Work A", "Author A", 2020, "10.0000/ref.A")
            db_manager.add_reference(paper_id_1, "Related Work B", "Author B, Author C", 2019, is_in_library=True) # Assume this one is in library

        # 10. Get references for a paper
        paper1_references = db_manager.get_references_for_paper(paper_id_1)
        logger.info(f"References for Paper 1 (ID: {paper_id_1}):")
        for ref in paper1_references:
            logger.info(f"  - {ref['cited_title']} ({ref['cited_year']}), In Library: {bool(ref['is_in_library'])}")

        # 11. Update summary
        if paper_id_1:
            db_manager.update_paper_summary(paper_id_1, "This is a concise summary of Paper 1, generated by the agent.")
            updated_paper = db_manager.get_paper(paper_id_1)
            logger.info(f"Paper 1 summary status: is_summarized={bool(updated_paper['is_summarized'])}, summary_text='{updated_paper['summary_text'][:50]}...'")

        # 12. Get all papers
        all_papers = db_manager.get_all_papers()
        logger.info(f"\nAll Papers in DB ({len(all_papers)} total):")
        for p in all_papers:
            logger.info(f"  - {p['title']} (ID: {p['id']})")

        # 13. Delete a paper (and observe cascade delete)
        if paper_id_2:
            logger.info(f"\nAttempting to delete Paper 2 (ID: {paper_id_2})...")
            db_manager.delete_paper(paper_id_2)
            deleted_paper = db_manager.get_paper(paper_id_2)
            if not deleted_paper:
                logger.info(f"Paper 2 (ID: {paper_id_2}) successfully deleted.")
            else:
                logger.error(f"Paper 2 (ID: {paper_id_2}) still exists after deletion attempt.")

            # Check if associated sections/tags were also deleted
            paper2_sections_after_delete = db_manager.get_sections_for_paper(paper_id_2)
            paper2_tags_after_delete = db_manager.get_tags_for_paper(paper_id_2)
            logger.info(f"Sections for Paper 2 after delete: {len(paper2_sections_after_delete)}")
            logger.info(f"Tags for Paper 2 after delete: {len(paper2_tags_after_delete)}")


    except Exception as e:
        logger.critical(f"An error occurred during DBManager initialization or testing: {e}", exc_info=True)

    logger.info("Paper Agent started successfully. (CLI not yet implemented)")


if __name__ == "__main__":
    main()

