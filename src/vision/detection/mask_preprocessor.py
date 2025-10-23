"""
Mask Preprocessor - Adaptive binarization and morphological cleanup.

Preprocesses motion masks for improved dart detection with:
- Adaptive Otsu thresholding
- Morphological operations (open → close)
- Small blob removal
- Quality checks
"""

import cv2
import numpy as np
import logging
from typing import Optional


logger = logging.getLogger(__name__)


class MaskPreprocessor:
    """
    Preprocess motion masks for dart detection.

    Features:
    - Adaptive binarization with Otsu + bias
    - Morphological noise removal
    - Small blob filtering
    - Quality validation
    """

    def __init__(
        self,
        adaptive: bool = True,
        otsu_bias: int = 10,
        open_kernel_size: int = 5,
        close_kernel_size: int = 9,
        min_blob_area: int = 24,
        min_white_fraction: float = 0.015,
    ):
        """
        Initialize mask preprocessor.

        Args:
            adaptive: Enable adaptive Otsu thresholding
            otsu_bias: Bias added to Otsu threshold
            open_kernel_size: Kernel size for morphological opening
            close_kernel_size: Kernel size for morphological closing
            min_blob_area: Minimum blob area to keep (pixels)
            min_white_fraction: Minimum white pixel fraction for valid mask
        """
        self.adaptive = adaptive
        self.otsu_bias = otsu_bias
        self.open_kernel_size = open_kernel_size | 1  # Ensure odd
        self.close_kernel_size = close_kernel_size | 1  # Ensure odd
        self.min_blob_area = min_blob_area
        self.min_white_fraction = min_white_fraction

        # Create kernels
        self._update_kernels()

        # Statistics
        self.stats = {
            "masks_processed": 0,
            "blobs_removed": 0,
            "rejected_empty": 0,
        }

    def _update_kernels(self):
        """Update morphological kernels"""
        self.open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.open_kernel_size, self.open_kernel_size)
        )
        self.close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.close_kernel_size, self.close_kernel_size)
        )

    def preprocess(self, motion_mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Preprocess motion mask.

        Args:
            motion_mask: Input motion mask (grayscale or binary)

        Returns:
            Preprocessed binary mask or None if quality check fails
        """
        self.stats["masks_processed"] += 1

        # Convert to grayscale if needed
        if motion_mask.ndim == 3:
            mask = cv2.cvtColor(motion_mask, cv2.COLOR_BGR2GRAY)
        else:
            mask = motion_mask.copy()

        # Adaptive binarization
        if self.adaptive:
            mask = self._adaptive_binarize(mask)
        else:
            # Simple threshold
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Morphological operations
        mask = self._morphological_cleanup(mask)

        # Remove small blobs
        mask = self._remove_small_blobs(mask)

        # Quality check
        if not self._quality_check(mask):
            self.stats["rejected_empty"] += 1
            return None

        return mask

    def _adaptive_binarize(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply adaptive Otsu thresholding.

        Args:
            mask: Input grayscale mask

        Returns:
            Binary mask
        """
        # Otsu threshold
        thresh_val, _ = cv2.threshold(
            mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Apply bias
        thresh_val = int(np.clip(thresh_val + self.otsu_bias, 0, 255))

        # Threshold with adjusted value
        _, binary = cv2.threshold(mask, thresh_val, 255, cv2.THRESH_BINARY)

        return binary

    def _morphological_cleanup(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations for cleanup.

        Args:
            mask: Input binary mask

        Returns:
            Cleaned binary mask
        """
        # Opening: remove noise
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel)

        # Closing: fill holes
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.close_kernel)

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
        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_blob_area:
                output[labels == i] = 255
            else:
                self.stats["blobs_removed"] += 1

        return output

    def _quality_check(self, mask: np.ndarray) -> bool:
        """
        Check if mask has sufficient content.

        Args:
            mask: Binary mask

        Returns:
            True if mask passes quality check
        """
        # Count white pixels
        white_pixels = cv2.countNonZero(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        white_fraction = white_pixels / total_pixels

        # Check minimum white fraction
        return white_fraction >= self.min_white_fraction

    def update_parameters(
        self,
        adaptive: Optional[bool] = None,
        otsu_bias: Optional[int] = None,
        open_kernel_size: Optional[int] = None,
        close_kernel_size: Optional[int] = None,
    ):
        """
        Update preprocessor parameters.

        Args:
            adaptive: Enable/disable adaptive thresholding
            otsu_bias: New Otsu bias value
            open_kernel_size: New opening kernel size
            close_kernel_size: New closing kernel size
        """
        if adaptive is not None:
            self.adaptive = adaptive

        if otsu_bias is not None:
            self.otsu_bias = otsu_bias

        kernel_changed = False
        if open_kernel_size is not None:
            self.open_kernel_size = open_kernel_size | 1
            kernel_changed = True

        if close_kernel_size is not None:
            self.close_kernel_size = close_kernel_size | 1
            kernel_changed = True

        if kernel_changed:
            self._update_kernels()

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "masks_processed": 0,
            "blobs_removed": 0,
            "rejected_empty": 0,
        }

    def get_stats(self) -> dict:
        """Get current statistics"""
        return self.stats.copy()
