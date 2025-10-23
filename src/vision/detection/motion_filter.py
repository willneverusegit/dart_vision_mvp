"""
Motion Filter - Morphological operations and mask cleanup.

Provides noise reduction and blob cleanup for motion masks
using morphological operations and connected component analysis.
"""

import cv2
import numpy as np
import logging
from typing import Tuple


logger = logging.getLogger(__name__)


class MotionFilter:
    """
    Filter and clean motion masks.

    Features:
    - Morphological operations (open/close)
    - Small blob removal
    - Configurable kernel sizes
    - Statistics tracking
    """

    def __init__(
        self,
        morph_kernel_size: int = 3,
        min_blob_area: int = 100,
    ):
        """
        Initialize motion filter.

        Args:
            morph_kernel_size: Size of morphological kernel (must be odd)
            min_blob_area: Minimum blob area to keep (pixels)
        """
        self.morph_kernel_size = morph_kernel_size | 1  # Ensure odd
        self.min_blob_area = min_blob_area

        # Create morphological kernel
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size),
        )

        # Statistics
        self.stats = {
            "blobs_removed": 0,
            "total_blobs": 0,
        }

    def apply(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply filtering to motion mask.

        Args:
            mask: Input binary mask

        Returns:
            Filtered mask
        """
        # Morphological opening (remove noise)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)

        # Morphological closing (fill holes)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.kernel)

        # Remove small blobs
        if self.min_blob_area > 0:
            cleaned = self._remove_small_blobs(cleaned)

        return cleaned

    def _remove_small_blobs(self, mask: np.ndarray) -> np.ndarray:
        """
        Remove small connected components.

        Args:
            mask: Input binary mask

        Returns:
            Mask with small blobs removed
        """
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        # Create output mask
        output = np.zeros_like(mask)

        # Keep blobs above minimum area
        blobs_kept = 0
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_blob_area:
                output[labels == i] = 255
                blobs_kept += 1

        # Update stats
        self.stats["total_blobs"] += num_labels - 1  # Exclude background
        self.stats["blobs_removed"] += (num_labels - 1) - blobs_kept

        return output

    def update_kernel_size(self, size: int):
        """
        Update morphological kernel size.

        Args:
            size: New kernel size (will be made odd)
        """
        self.morph_kernel_size = size | 1
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size),
        )
        logger.debug(f"Updated kernel size to {self.morph_kernel_size}")

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "blobs_removed": 0,
            "total_blobs": 0,
        }

    def get_stats(self) -> dict:
        """Get current statistics"""
        return self.stats.copy()
