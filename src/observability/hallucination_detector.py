"""
Hallucination detection for RAG systems using semantic similarity.

Detects when LLM responses are not grounded in retrieved context by:
- Computing embeddings of context and response
- Calculating cosine similarity
- Deriving faithfulness and hallucination risk scores

Based on ReadyTensor Week 11 Lesson 1b: "What to Monitor in Agentic AI"
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HALLUCINATION_DETECTION_AVAILABLE = True
except ImportError:
    HALLUCINATION_DETECTION_AVAILABLE = False
    SentenceTransformer = None
    cosine_similarity = None

logger = logging.getLogger(__name__)


class HallucinationDetector:
    """
    Semantic similarity-based hallucination detector for RAG systems.

    Uses sentence embeddings to measure how well the LLM response
    is grounded in the retrieved context chunks.

    Metrics:
    - Faithfulness Score (0-1): How well response aligns with context
    - Hallucination Risk (0-1): Inverse of faithfulness (1 - faithfulness)
    - Max Similarity: Best alignment with any context chunk
    - Avg Similarity: Average alignment across all chunks

    Example:
        detector = HallucinationDetector()

        result = detector.calculate_faithfulness_score(
            retrieved_contexts=["AI is...", "ML is..."],
            llm_response="Artificial intelligence is a field..."
        )

        print(f"Faithfulness: {result['faithfulness_score']:.2f}")
        print(f"Hallucination Risk: {result['hallucination_risk']:.2f}")
    """

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        enabled: bool = True,
        hallucination_threshold: float = 0.3
    ):
        """
        Initialize hallucination detector.

        Args:
            model_name: Sentence-Transformers model name
                       'all-MiniLM-L6-v2' is fast and accurate (default)
            enabled: Whether to enable hallucination detection
            hallucination_threshold: Threshold for warning (0-1)
                                   Higher values = more strict
        """
        self.enabled = enabled and HALLUCINATION_DETECTION_AVAILABLE
        self.model: Optional[Any] = None
        self.hallucination_threshold = hallucination_threshold

        if not self.enabled:
            if not HALLUCINATION_DETECTION_AVAILABLE:
                logger.warning(
                    "sentence-transformers or scikit-learn not installed. "
                    "Hallucination detection disabled."
                )
            else:
                logger.info("Hallucination detection disabled by configuration.")
            return

        try:
            # Load sentence transformer model
            self.model = SentenceTransformer(model_name)
            logger.info(f"Hallucination detector initialized (model: {model_name})")
        except Exception as e:
            logger.error(f"Failed to load hallucination detection model: {e}")
            self.enabled = False
            self.model = None

    def calculate_faithfulness_score(
        self,
        retrieved_contexts: List[str],
        llm_response: str
    ) -> Dict[str, float]:
        """
        Calculate faithfulness score using semantic similarity.

        The faithfulness score measures how well the LLM response
        is grounded in the retrieved context. Higher scores indicate
        better alignment.

        Args:
            retrieved_contexts: List of context chunks from vector DB
            llm_response: LLM-generated response text

        Returns:
            Dict with:
            - faithfulness_score: Main score (0-1, higher is better)
            - hallucination_risk: Inverse score (0-1, lower is better)
            - max_similarity: Best similarity with any chunk
            - avg_similarity: Average similarity across chunks
            - is_hallucination: Boolean warning flag

        Example:
            result = detector.calculate_faithfulness_score(
                retrieved_contexts=["Python is a programming language."],
                llm_response="Python is used for web development."
            )
            # result['faithfulness_score'] ≈ 0.75 (good alignment)
            # result['hallucination_risk'] ≈ 0.25 (low risk)
        """
        if not self.enabled or not self.model:
            # Return neutral scores if disabled
            return {
                'faithfulness_score': 0.5,
                'hallucination_risk': 0.5,
                'max_similarity': 0.5,
                'avg_similarity': 0.5,
                'is_hallucination': False
            }

        # Handle edge cases
        if not retrieved_contexts or not llm_response.strip():
            logger.warning("Empty contexts or response for hallucination detection")
            return {
                'faithfulness_score': 0.0,
                'hallucination_risk': 1.0,
                'max_similarity': 0.0,
                'avg_similarity': 0.0,
                'is_hallucination': True
            }

        try:
            # Encode context chunks
            context_embeddings = self.model.encode(
                retrieved_contexts,
                convert_to_numpy=True
            )

            # Encode LLM response
            response_embedding = self.model.encode(
                [llm_response],
                convert_to_numpy=True
            )

            # Calculate cosine similarities
            similarities = cosine_similarity(
                response_embedding,
                context_embeddings
            )[0]

            # Compute metrics
            max_sim = float(np.max(similarities))
            avg_sim = float(np.mean(similarities))

            # Faithfulness = how well response aligns with context
            faithfulness_score = max_sim

            # Hallucination risk = inverse of faithfulness
            hallucination_risk = 1.0 - faithfulness_score

            # Flag as potential hallucination if risk is high
            is_hallucination = hallucination_risk > self.hallucination_threshold

            result = {
                'faithfulness_score': faithfulness_score,
                'hallucination_risk': hallucination_risk,
                'max_similarity': max_sim,
                'avg_similarity': avg_sim,
                'is_hallucination': is_hallucination
            }

            if is_hallucination:
                logger.warning(
                    f"Potential hallucination detected! "
                    f"Risk: {hallucination_risk:.2f}, "
                    f"Faithfulness: {faithfulness_score:.2f}"
                )

            return result

        except Exception as e:
            logger.error(f"Error calculating faithfulness score: {e}")
            return {
                'faithfulness_score': 0.0,
                'hallucination_risk': 1.0,
                'max_similarity': 0.0,
                'avg_similarity': 0.0,
                'is_hallucination': True,
                'error': str(e)
            }

    def analyze_response_quality(
        self,
        retrieved_contexts: List[str],
        llm_response: str,
        include_details: bool = False
    ) -> Dict[str, Any]:
        """
        Comprehensive response quality analysis.

        Extends faithfulness calculation with additional insights:
        - Per-chunk similarity breakdown
        - Context coverage analysis
        - Confidence assessment

        Args:
            retrieved_contexts: Retrieved context chunks
            llm_response: LLM response to analyze
            include_details: Whether to include per-chunk details

        Returns:
            Dict with faithfulness metrics and optional details
        """
        # Get base faithfulness score
        result = self.calculate_faithfulness_score(
            retrieved_contexts,
            llm_response
        )

        if not include_details or not self.enabled:
            return result

        try:
            # Calculate per-chunk similarities
            context_embeddings = self.model.encode(
                retrieved_contexts,
                convert_to_numpy=True
            )
            response_embedding = self.model.encode(
                [llm_response],
                convert_to_numpy=True
            )
            similarities = cosine_similarity(
                response_embedding,
                context_embeddings
            )[0]

            # Find most relevant chunks
            chunk_scores = [
                {
                    'chunk_index': i,
                    'similarity': float(sim),
                    'text_preview': context[:100] + "..." if len(context) > 100 else context
                }
                for i, (sim, context) in enumerate(zip(similarities, retrieved_contexts))
            ]

            # Sort by similarity
            chunk_scores.sort(key=lambda x: x['similarity'], reverse=True)

            # Add details to result
            result['chunk_analysis'] = {
                'total_chunks': len(retrieved_contexts),
                'top_3_chunks': chunk_scores[:3],
                'similarity_distribution': {
                    'min': float(np.min(similarities)),
                    'max': float(np.max(similarities)),
                    'std': float(np.std(similarities))
                }
            }

            return result

        except Exception as e:
            logger.error(f"Error in detailed analysis: {e}")
            return result

    def batch_calculate_faithfulness(
        self,
        context_response_pairs: List[Dict[str, Any]]
    ) -> List[Dict[str, float]]:
        """
        Batch process multiple context-response pairs.

        Efficient for analyzing multiple queries at once.

        Args:
            context_response_pairs: List of dicts with:
                - 'contexts': List[str]
                - 'response': str

        Returns:
            List of faithfulness score dicts

        Example:
            pairs = [
                {
                    'contexts': ["Python is..."],
                    'response': "Python is a language"
                },
                {
                    'contexts': ["AI is..."],
                    'response': "AI involves machine learning"
                }
            ]
            results = detector.batch_calculate_faithfulness(pairs)
        """
        results = []

        for pair in context_response_pairs:
            result = self.calculate_faithfulness_score(
                retrieved_contexts=pair.get('contexts', []),
                llm_response=pair.get('response', '')
            )
            results.append(result)

        return results
