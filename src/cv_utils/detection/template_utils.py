"""
Utility functions for template matching.
"""

from pathlib import Path

import numpy as np

from ..loaders import ImageLoader
from .matchers import TemplateMatcher
from .pipelines.template_pipeline import TemplateMatchingPipeline
from .template_detector import TemplateDetector


def compare_methods(
    image_path: str | Path,
    template_path: str | Path,
    show_score: bool = True,
) -> tuple[list[np.ndarray], list[str], list[dict]]:
    """
    Compare all template matching methods on the same image.

    Tests all 6 OpenCV template matching methods with their default thresholds.

    Args:
        image_path: Path to the source image
        template_path: Path to the template to find
        show_score: Whether to show match score on images (default: True)

    Returns:
        tuple: (list of result images, list of titles, list of metadata dicts)

    Example:
        >>> from cv_utils.detection.template_utils import compare_methods
        >>> from cv_utils.visualisation import display_grid
        >>>
        >>> results, titles, metadata = compare_methods(
        ...     image_path="screenshot.jpg",
        ...     template_path="logo.png"
        ... )
        >>> display_grid(results, titles=titles, max_cols=3)
    """
    # Load image once as RGB (same as ImageProcessor and TemplateDetector)
    image = ImageLoader.load(image_path, "rgb")
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Use all methods from TemplateMatcher
    methods = TemplateMatcher.SUPPORTED_METHODS

    # Use default thresholds from TemplateMatcher
    thresholds = TemplateMatcher.DEFAULT_THRESHOLDS

    results = []
    titles = []
    metadata_list = []

    for method_str in methods:
        # Extract clean name for display (without "cv2." prefix)
        name = method_str.replace("cv2.", "")

        # Get threshold for this method (using method_str with cv2. prefix)
        threshold = thresholds.get(method_str, None)
        detector = TemplateDetector(
            template=template_path, method=method_str, threshold=threshold
        )

        pipeline = TemplateMatchingPipeline(detector=detector, show_score=show_score)

        # Process image directly
        result = pipeline.process(image)

        # Collect results
        results.append(result)
        meta = pipeline.metadata
        metadata_list.append(meta)

        # Create title with status
        match_status = "[MATCH]" if meta["match_found"] else "[NO MATCH]"
        score = f"{meta['score']:.3f}" if meta["score"] is not None else "N/A"
        titles.append(f"{name}\n{match_status} (score: {score})")

    return results, titles, metadata_list
