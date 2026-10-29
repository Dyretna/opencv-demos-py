"""
Base classes and interfaces for all detection components.

This module defines the abstract interfaces that all detection components
across cv_utils should implement, enabling modular and composable systems.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Detector(ABC):
    """
    Base interface for any detector component.

    A detector takes an image and returns detection data.
    Examples: face detector, line detector, object detector, etc.
    Should be stateless and side-effect free.
    """

    @abstractmethod
    def detect(self, image: np.ndarray) -> Any:
        """
        Detect features in an image.

        Args:
            image: Input image as numpy array

        Returns:
            Detection data (type varies by implementation)
        """
        pass


class Pipeline(ABC):
    """
    Base interface for detection pipelines.

    A pipeline orchestrates multiple detectors and processors
    to produce a final detection result.

    All pipelines should provide metadata about the last processed frame
    via the metadata property.
    """

    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Run the full detection pipeline on an image.

        Args:
            image: Input image

        Returns:
            Processed image as numpy array
        """
        pass

    @property
    def metadata(self) -> dict[str, Any]:
        """
        Get metadata from the last processed frame.

        Returns:
            Dictionary with pipeline-specific metadata.
            Common keys include:
            - 'count': Number of objects/features detected
            - 'detections': Raw detection data
            - 'processing_time': Time taken to process

        Note:
            Subclasses should override this to provide meaningful metadata.
            Default implementation returns empty dict.
        """
        return {}

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any]) -> "Pipeline":
        """
        Create a new pipeline instance from a configuration dictionary.

        Args:
            config: Dictionary of pipeline parameters.
            For partial updates, merge with current config.

        Returns:
            Instance of Pipeline
        """
        pass

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """
        Return the current configuration of the pipeline as a dictionary.

        Returns:
            Dictionary of pipeline parameters
        """
        pass
