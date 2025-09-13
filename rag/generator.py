# paper_agent/rag/generator.py

from typing import List, Dict, Any, Optional

from llm.llm_interface import LLMInterface
from utils.logger import logger


class Generator:
    """
    Generates an answer to a user query based on retrieved document sections
    using a Large Language Model.
    """

    def __init__(self, llm_interface: LLMInterface):
        self.llm = llm_interface
        logger.info("Generator initialized.")

    def generate_answer(self, query: str, retrieved_sections: List[Dict[str, Any]]) -> Optional[str]:
        """
        Constructs a prompt with the query and retrieved context, then
        uses the LLM to generate an answer.

        Args:
            query (str): The user's original query.
            retrieved_sections (List[Dict[str, Any]]): A list of relevant sections
                                                      (each containing 'content', 'paper_title', 'section_title').

        Returns:
            Optional[str]: The LLM-generated answer, or None if generation fails.
        """
        if not retrieved_sections:
            return "I couldn't find any relevant information in your papers to answer that question."

        context_parts = []
        for i, section in enumerate(retrieved_sections):
            context_parts.append(f"--- Document {i+1} ---")
            context_parts.append(f"Paper Title: {section.get('paper_title', 'N/A')}")
            context_parts.append(f"Section Title: {section.get('section_title', 'N/A')}")
            context_parts.append(f"Content:\n{section['content']}")
            context_parts.append("\n")

        context = "\n".join(context_parts)

        prompt = f"""
        You are an AI assistant that answers questions based on the provided academic paper excerpts.
        Your goal is to provide a concise and accurate answer to the user's question,
        strictly using only the information available in the "Context" sections below.
        If the answer cannot be found in the provided context, state that you don't have enough information.
        Do NOT make up any information.

        Context:
        {context}

        Question: {query}

        Answer:
        """

        logger.debug(f"Sending prompt to LLM (first 500 chars): {prompt[:500]}...")
        answer = self.llm.generate_text(prompt, max_tokens=500, temperature=0.2)  # Lower temperature for factual answers

        if answer:
            logger.info(f"LLM generated answer for query: '{query[:50]}...'")
            return answer
        else:
            logger.error(f"LLM failed to generate an answer for query: '{query}'")
            return "I apologize, but I encountered an error while trying to generate an answer."
