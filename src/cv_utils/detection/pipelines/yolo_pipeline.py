from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ...drawing import draw_labeled_box
from ..base import Pipeline

CATEGORY_MAP = {
    # --- Human ---
    "person": "human",
    # --- Vehicles ---
    "bicycle": "vehicles",
    "car": "vehicles",
    "motorbike": "vehicles",
    "aeroplane": "vehicles",
    "bus": "vehicles",
    "train": "vehicles",
    "truck": "vehicles",
    "boat": "vehicles",
    # --- Animals ---
    "bird": "animals",
    "cat": "animals",
    "dog": "animals",
    "horse": "animals",
    "sheep": "animals",
    "cow": "animals",
    "elephant": "animals",
    "bear": "animals",
    "zebra": "animals",
    "giraffe": "animals",
    # --- Sport & Lifestyle ---
    "backpack": "sport_lifestyle",
    "umbrella": "sport_lifestyle",
    "handbag": "sport_lifestyle",
    "tie": "sport_lifestyle",
    "suitcase": "sport_lifestyle",
    "frisbee": "sport_lifestyle",
    "skis": "sport_lifestyle",
    "snowboard": "sport_lifestyle",
    "sports ball": "sport_lifestyle",
    "kite": "sport_lifestyle",
    "baseball bat": "sport_lifestyle",
    "baseball glove": "sport_lifestyle",
    "skateboard": "sport_lifestyle",
    "surfboard": "sport_lifestyle",
    "tennis racket": "sport_lifestyle",
    # --- Kitchen stuff ---
    "bottle": "kitchen",
    "wine glass": "kitchen",
    "cup": "kitchen",
    "fork": "kitchen",
    "knife": "kitchen",
    "spoon": "kitchen",
    "bowl": "kitchen",
    # --- Food ---
    "banana": "food",
    "apple": "food",
    "sandwich": "food",
    "orange": "food",
    "broccoli": "food",
    "carrot": "food",
    "hot dog": "food",
    "pizza": "food",
    "donut": "food",
    "cake": "food",
    # in house things
    "chair": "in_house_things",
    "sofa": "in_house_things",
    "pottedplant": "in_house_things",
    "bed": "in_house_things",
    "diningtable": "in_house_things",
    "toilet": "in_house_things",
    "tvmonitor": "in_house_things",
    "laptop": "in_house_things",
    "mouse": "in_house_things",
    "remote": "in_house_things",
    "keyboard": "in_house_things",
    "cell phone": "in_house_things",
    "microwave": "in_house_things",
    "oven": "in_house_things",
    "toaster": "in_house_things",
    "refrigerator book": "in_house_things",
    "clock": "in_house_things",
    "vase": "in_house_things",
    "scissor": "in_house_things",
    "teddy bear": "in_house_things",
    "hair drier": "in_house_things",
    # misc
    "toothbrush": "MISC",
}


class YOLOPipeline(Pipeline):
    CATEGORY_MAP = CATEGORY_MAP

    def __init__(
        self,
        config_file: str | Path,
        labels_file: str | Path,
        weights_file: str | Path,
        confidence_thres: float = 0.1,
        score_thres: float = 0.9,
        drawing_params: dict | None = None,
        crops_path: str | Path | None = None,
    ):
        self.config_file = config_file
        self.labels_file = labels_file
        self.weights_file = weights_file

        # output folder of crops. If None, no output
        if not isinstance(crops_path, (str | Path | None)):
            raise TypeError("crops_path must be of type: str, Path or None")
        self.crops_path = crops_path

        # get YOLO model from config file and weights
        self.model = cv2.dnn.readNetFromDarknet(
            str(self.config_file), str(self.weights_file)
        )

        #################### CUDA #######################

        # try:
        #     self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        #     self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        #     print("Using CUDA backend")
        # except Exception as e:
        #     print("CUDA not available, falling back to cpu:", e)
        #     self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        #     self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        ##################################################

        # link unconnected output layers to
        # unconnected correct layer names
        # eg. ('yolo_139', 'yolo_150', 'yolo_161')
        self.linked_names = self._link_names()

        with open(self.labels_file) as f:
            self.classes = f.read().strip().split("\n")

        # thresholds
        self.confidence_thres: float = confidence_thres  # 0 > thres < 1
        self.score_thres = score_thres
        self.non_max_suppression_thres = score_thres

        self.layer_output = None
        self.found_boxes = None

        if drawing_params is not None:
            self.drawing_params = drawing_params
        else:
            self.drawing_params = {
                "font_scale": 1,
                "color": (255, 255, 0),
                "thickness": 2,
            }

    def process(self, image):
        # boxes, confidences, classes
        # must reset for frame
        self.boxes = []
        self.confidence_scores = []
        self.classIDs = []

        self.height, self.width, self.color = image.shape
        # Model init
        blob = self._set_blob(image)

        self.model.setInput(blob)  # prepares the img to match input layer
        self.layer_output = self.model.forward(self.linked_names)

        self._set_confidence_boxes()
        self.found_boxes = cv2.dnn.NMSBoxes(
            bboxes=self.boxes,
            scores=self.confidence_scores,
            score_threshold=self.score_thres,
            nms_threshold=self.non_max_suppression_thres,
        )
        result = image.copy()
        self._draw_boxes(result)
        if self.crops_path is not None:
            self._save_crops(image)
        return result

    # TODO: these are taken from Pipeline ABC,
    # rewrite when the rest is functioning

    @property
    def metadata(self) -> dict[str, Any]:
        """Get metadata from the last processed frame."""
        return {
            "count": len(self.found_boxes) if self.found_boxes is not None else 0,
            "labels": [self.classes[i] for i in self.classIDs],
            "confidences": self.confidence_scores,
            "boxes": self.boxes,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Pipeline":
        """
        Create a new pipeline instance from a configuration dictionary.

        Args:
            config: Dictionary of pipeline parameters.
            For partial updates, merge with current config.

        Returns:
            Instance of Pipeline
        """
        return cls(**config)

    def get_config(self) -> dict[str, Any]:
        """
        Return the current configuration of the pipeline as a dictionary.

        Returns:
            Dictionary of pipeline parameters
        """
        return {
            "config_file": str(self.config_file),
            "labels_file": str(self.labels_file),
            "weights_file": str(self.weights_file),
            "confidence_thres": self.confidence_thres,
            "score_thres": self.score_thres,
            "drawing_params": self.drawing_params,
        }

    #   +---------------------+
    #   |   Private Helpers   |
    #   +---------------------+

    def _link_names(self):
        """Attached names of objects to models output layer.

        This is important because...
        """
        linked_names = self.model.getLayerNames()
        return [linked_names[i - 1] for i in self.model.getUnconnectedOutLayers()]

    def _set_blob(self, image: np.ndarray):
        """Sets the blob.

        Image = src image
        Scalefactor= 1/255 - because...
        size=(608, 608) bcs images based on trained model
        swap RB = True, because we send bgr
        crop = False, to not miss any object
        """
        return cv2.dnn.blobFromImage(
            image=image,
            scalefactor=1 / 255,
            size=(608, 608),
            swapRB=True,
            crop=False,
            # ddepth=1,
        )

    def _set_confidence_boxes(self):
        for output in self.layer_output:
            # print(output)
            for detected_box in output:
                # based on Docs:
                # detected_box[:5] == box geometries
                # detected_box[5:] == scores
                scores = detected_box[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > self.confidence_thres:
                    self._save_boxes_from_output(detected_box)
                    # grab confidence of this box
                    self.confidence_scores.append(float(confidence))
                    # grab Class ID
                    self.classIDs.append(classID)
                    # print(classID)

    def _save_boxes_from_output(self, detected_box):
        # We need to find the box around this object
        box = detected_box[:4] * np.array(
            [self.width, self.height, self.width, self.height]
        )
        (center_x, center_y, box_width, box_height) = box.astype("int")
        x = int((center_x - box_width / 2))  # x for top left
        y = int((center_y - box_height / 2))

        # grab box topleft corner coordinations + width + height
        self.boxes.append([x, y, int(box_width), int(box_height)])

    def _save_crops(self, image):
        # found_boxes is tuple if empty, tuple or 0
        if self.found_boxes is None:
            return

        if isinstance(self.found_boxes, tuple) or len(self.found_boxes) == 0:
            return

        for idx in self.found_boxes.flatten():
            label = self.classes[self.classIDs[idx]]
            category = self.CATEGORY_MAP.get(label)
            if category is None:
                continue

            confidence = self.confidence_scores[idx]
            if confidence < self.score_thres:
                continue

            x, y, w, h = self.boxes[idx]
            crop = image[y : y + h, x : x + w]
            if self.crops_path is not None:
                # never happens, since crops path is flag
                # for saving crop. nevertheless, for the future...
                out_dir = Path(self.crops_path) / category
            else:
                out_dir = Path("YOLO_output") / category
            out_dir.mkdir(parents=True, exist_ok=True)

            filename = out_dir / f"{label}_{idx}.jpg"
            cv2.imwrite(str(filename), crop)

    def _draw_boxes(self, img: np.ndarray):
        if len(self.found_boxes) > 0:
            for idx in self.found_boxes.flatten():
                (x, y) = (self.boxes[idx][0], self.boxes[idx][1])
                (width, height) = (self.boxes[idx][2], self.boxes[idx][3])

                tl = (x, y)
                br = (x + width, y + height)
                label = self.classes[self.classIDs[idx]]
                confidence = self.confidence_scores[idx]

                draw_labeled_box(
                    img=img,
                    tl=tl,
                    br=br,
                    text=f"{label}, {confidence:.2f}",
                    fontscale=self.drawing_params["font_scale"],
                    color=self.drawing_params["color"],
                    thickness=self.drawing_params["thickness"],
                )
