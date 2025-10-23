"""
Background Subtractor - MOG2 background subtraction wrapper.

Provides clean interface to OpenCV's MOG2 background subtractor
with configurable parameters and shadow detection.
"""

import cv2
import numpy as np
import logging
from typing import Optional


logger = logging.getLogger(__name__)


class BackgroundSubtractor:
    """
    Wrapper for MOG2 background subtraction.

    Features:
    - Configurable variance threshold
    - Shadow detection
    - Adaptive learning rate
    - Statistics tracking
    """

    def __init__(
        self,
        var_threshold: int = 50,
        detect_shadows: bool = True,
        history: int = 500,
        learning_rate: float = -1.0,
    ):
        """
        Initialize background subtractor.

        Args:
            var_threshold: MOG2 variance threshold (lower = more sensitive)
            detect_shadows: Enable shadow detection
            history: Number of frames for background model
            learning_rate: Learning rate (-1 for automatic)
        """
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.history = history
        self.learning_rate = learning_rate

        # Create MOG2 subtractor
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=detect_shadows,
            varThreshold=var_threshold,
            history=history,
        )

        # Statistics
        self.frames_processed = 0

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply background subtraction to frame.

        Args:
            frame: Input BGR frame

        Returns:
            Foreground mask (grayscale)
        """
        # Apply MOG2
        fg_mask = self.subtractor.apply(frame, learningRate=self.learning_rate)

        # Remove shadows (if detected, they have value 127)
        if self.detect_shadows:
            fg_mask[fg_mask == 127] = 0

        self.frames_processed += 1

        return fg_mask

    def update_parameters(
        self,
        var_threshold: Optional[int] = None,
        detect_shadows: Optional[bool] = None,
        learning_rate: Optional[float] = None,
    ):
        """
        Update subtractor parameters.

        Note: Some parameters require recreating the subtractor.

        Args:
            var_threshold: New variance threshold
            detect_shadows: Enable/disable shadow detection
            learning_rate: New learning rate
        """
        needs_recreation = False

        if var_threshold is not None and var_threshold != self.var_threshold:
            self.var_threshold = var_threshold
            needs_recreation = True

        if detect_shadows is not None and detect_shadows != self.detect_shadows:
            self.detect_shadows = detect_shadows
            needs_recreation = True

        if learning_rate is not None:
            self.learning_rate = learning_rate

        # Recreate subtractor if needed
        if needs_recreation:
            self.subtractor = cv2.createBackgroundSubtractorMOG2(
                detectShadows=self.detect_shadows,
                varThreshold=self.var_threshold,
                history=self.history,
            )
            logger.debug(f"Recreated MOG2 with var_threshold={self.var_threshold}")

    def reset(self):
        """Reset background model"""
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=self.detect_shadows,
            varThreshold=self.var_threshold,
            history=self.history,
        )
        self.frames_processed = 0
        logger.info("Background model reset")

    def get_background_image(self) -> np.ndarray:
        """
        Get current background model image.

        Returns:
            Background image
        """
        return self.subtractor.getBackgroundImage()
