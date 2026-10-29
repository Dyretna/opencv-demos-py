import cv2
import numpy as np


class TemplateMatcher:
    SUPPORTED_METHODS = [
        "cv2.TM_CCOEFF",
        "cv2.TM_CCOEFF_NORMED",
        "cv2.TM_CCORR",
        "cv2.TM_CCORR_NORMED",
        "cv2.TM_SQDIFF",
        "cv2.TM_SQDIFF_NORMED",
    ]

    DEFAULT_THRESHOLDS = {
        "cv2.TM_CCOEFF": 10000000,
        "cv2.TM_CCOEFF_NORMED": 0.7,
        "cv2.TM_CCORR": 50000000,
        "cv2.TM_CCORR_NORMED": 0.8,
        "cv2.TM_SQDIFF": 5000000,
        "cv2.TM_SQDIFF_NORMED": 0.3,
    }

    def __init__(self):
        self.method = None
        self.image_shape, self.mask_shape = None, None
        self.result, self.result_shape = None, None
        self.minVal, self.maxVal = None, None
        self.minLoc, self.maxLoc = None, None
        self.top_left, self.bottom_right = None, None

    def match(self, image: np.ndarray, mask: np.ndarray, method: str):
        self._validate_method(method)
        self.method = method

        self.image_shape = image.shape
        self.mask_shape = mask.shape

        self.result = self._match_template(image, mask, method)
        self.result_shape = self.result.shape

        (self.minVal, self.maxVal, self.minLoc, self.maxLoc) = cv2.minMaxLoc(
            self.result
        )

        self.top_left, self.bottom_right = self._set_corners()

    def get_result(self):
        return self.result

    def is_good_match(self, threshold: float | None) -> bool:
        """Check if the template match quality exceeds the threshold."""
        # Set default threshold based on method
        if threshold is None:
            threshold = self.DEFAULT_THRESHOLDS[self.method]

        if self.method in ["cv2.TM_SQDIFF", "cv2.TM_SQDIFF_NORMED"]:
            return self.minVal < threshold
        else:
            return self.maxVal > threshold

    def get_match_score(self) -> float:
        """Get the match quality score."""
        if self.method in ["cv2.TM_SQDIFF", "cv2.TM_SQDIFF_NORMED"]:
            return self.minVal
        else:
            return self.maxVal

    def __str__(self):
        info = [
            f"{key}: {value}" for key, value in self.__dict__.items() if key != "result"
        ]
        return "\n".join(info) + "\nFor result (array) use '.get_result()'"

    # --- private helpers ---

    def _validate_method(self, method: str) -> None:
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"method must be in {self.SUPPORTED_METHODS}")

    def _match_template(self, image, mask, method) -> np.ndarray:
        return cv2.matchTemplate(image, mask, eval(method))

    def _set_corners(self) -> tuple[tuple[int, int]]:
        # Handle both grayscale (2D) and color (3D) images
        if len(self.mask_shape) == 3:
            height, width, _ = self.mask_shape
        else:
            height, width = self.mask_shape

        if self.method in ["cv2.TM_SQDIFF", "cv2.TM_SQDIFF_NORMED"]:
            top_left = self.minLoc
        else:
            top_left = self.maxLoc
        bottom_right = (top_left[0] + width, top_left[1] + height)
        return top_left, bottom_right
