from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


class SimilarityService:
    """Service responsible for computing semantic similarity scores."""

    def cosine_similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        left_vector = np.asarray(left, dtype=np.float32).reshape(1, -1)
        right_vector = np.asarray(right, dtype=np.float32).reshape(1, -1)
        numerator = float(np.dot(left_vector[0], right_vector[0]))
        denom_left = float(np.linalg.norm(left_vector[0]))
        denom_right = float(np.linalg.norm(right_vector[0]))
        denominator = denom_left * denom_right
        if denominator == 0:
            logger.warning("Cosine similarity denominator is zero left_norm=%s right_norm=%s", denom_left, denom_right)
            return 0.0
        similarity = max(0.0, min(1.0, numerator / denominator))
        logger.debug("Cosine similarity numerator=%s denominator=%s similarity=%s", numerator, denominator, similarity)
        return similarity
