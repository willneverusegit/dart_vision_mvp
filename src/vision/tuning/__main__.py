"""
Entry point for parameter tuning tool.

Usage:
    python -m src.vision.tuning --video path/to/video.mp4
    python -m src.vision.tuning --webcam 0
"""

from .parameter_tuner import main

if __name__ == "__main__":
    main()
