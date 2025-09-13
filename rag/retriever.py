# paper_agent/rag/retriever.py

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from database.db_manager import DBManager
from embeddings.embedding_model import EmbeddingModel
from utils.logger import logger


class Retriever:
    """
    Retrieves relevant document sections based on a query using vector similarity search.
    """

    def __init__(self, db_manager: DBManager, embedding_model: EmbeddingModel):
        self.db_manager = db_manager
        self.embedding_model = embedding_model
        logger.info("Retriever initialized.")

    def _get_all_section_embeddings(self) -> List[Dict[str, Any]]:
        """
        Retrieves all section IDs, content, and their embeddings from the database.
        Returns a list of dictionaries, each containing 'id', 'content', and 'embedding'.
        """
        all_sections_data = []
        # Get all paper IDs first
        all_papers = self.db_manager.get_all_papers()
        if not all_papers:
            logger.warning("No papers found in the database for retrieval.")
            return []

        for paper in all_papers:
            paper_id = paper["id"]
            sections = self.db_manager.get_sections_for_paper(paper_id)
            for section in sections:
                if section["embedding"] is not None:
                    all_sections_data.append(
                        {
                            "id": section["id"],
                            "paper_id": paper_id,  # Include paper_id for context
                            "content": section["content"],
                            "page_number": section["page_number"],
                            "section_title": section["section_title"],
                            "embedding": section["embedding"],
                        }
                    )
        logger.debug(f"Loaded {len(all_sections_data)} section embeddings from database.")
        return all_sections_data

    def retrieve_relevant_sections(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Generates an embedding for the query and finds the top_k most similar sections.

        Args:
            query (str): The user's query string.
            top_k (int): The number of top relevant sections to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a relevant section
                                   including its content, paper_id, and similarity score.
        """
        query_embedding = self.embedding_model.get_embedding(query)
        if query_embedding is None:
            logger.error("Failed to generate embedding for the query.")
            return []

        all_sections = self._get_all_section_embeddings()
        if not all_sections:
            return []

        # Extract embeddings and section data
        section_embeddings = np.array([s["embedding"] for s in all_sections])
        section_data = [
            {
                "id": s["id"],
                "paper_id": s["paper_id"],
                "content": s["content"],
                "page_number": s["page_number"],
                "section_title": s["section_title"],
            }
            for s in all_sections
        ]

        # Calculate cosine similarity between query and all section embeddings
        # Reshape query_embedding to (1, -1) for cosine_similarity
        similarities = cosine_similarity(query_embedding.reshape(1, -1), section_embeddings)[0]

        # Combine similarities with section data and sort
        scored_sections = []
        for i, similarity in enumerate(similarities):
            scored_sections.append({"score": similarity, **section_data[i]})

        # Sort by score in descending order and return top_k
        scored_sections.sort(key=lambda x: x["score"], reverse=True)

        # Add paper title to each retrieved section for better context
        final_results = []
        for sec in scored_sections[:top_k]:
            paper_info = self.db_manager.get_paper(sec["paper_id"])
            sec["paper_title"] = paper_info.get("title", "Unknown Paper") if paper_info else "Unknown Paper"
            final_results.append(sec)

        logger.info(f"Retrieved {len(final_results)} relevant sections for query: '{query[:50]}...'")
        for res in final_results:
            logger.debug(
                f"  - Score: {res['score']:.4f}, Paper: {res['paper_title']}, Section: {res['section_title']} (ID: {res['id']})"
            )

        return final_results
