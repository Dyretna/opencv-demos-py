import cv2
import matplotlib.pyplot as plt
import numpy as np


class HistogramEqualizers:
    @staticmethod
    def enhance_lightning_bgr(img_bgr) -> np.ndarray:
        """Equalizes with hsv. Takes and returns BGR."""
        # convert to hsv, and equalize
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
        # return a rgb - convert back from hsv
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def enhance_lightning_gs(img_gs) -> np.ndarray:
        return cv2.equalizeHist(img_gs)

    @staticmethod
    def get_gs_dist(img: np.ndarray):
        dist_dict = {
            "min": img.min(),
            "max": img.max(),
            "mean": int(img.mean()),
            "median": int(np.median(img)),
        }
        print("grayscale distribution:", "\n", "-" * 20)
        for keys, vals in dist_dict.items():
            print(f"{keys:<6}: {vals}")

        return dist_dict

    @staticmethod
    def plot_histogram_bgr(img):
        colors = ("b", "g", "r")
        for channel, color in enumerate(colors):
            hist = cv2.calcHist(
                [img],
                channels=[channel],
                mask=None,
                histSize=[256],
                ranges=[0, 256],
            )
            plt.plot(hist, color=color)
        plt.title("BGR Histograms")
        plt.show()

    @staticmethod
    def plot_histogram_rgb(img):
        colors = ("r", "g", "b")
        for channel, color in enumerate(colors):
            hist = cv2.calcHist(
                [img], channels=[channel], mask=None, histSize=[256], ranges=[0, 256]
            )
            plt.plot(hist, color=color)
        plt.title("RGB Histograms")
        plt.show()

    @staticmethod
    def plot_histogram_gs(img):
        hist = cv2.calcHist(
            [img], channels=[0], mask=None, histSize=[256], ranges=[0, 256]
        )
        plt.plot(hist)
        plt.title("Grayscale Histogram")
        plt.show()
