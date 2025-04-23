import matplotlib.pyplot as plt
import requests
from io import BytesIO
from datetime import datetime
from math import pi, cos
from typing import List, Dict
import numpy as np
import cv2
from PIL import Image

from tabulate import tabulate

plt.rcParams["image.cmap"] = "gray"


class Sun:
    def __init__(
        self,
        sun_img: Image.Image,
        date_time: datetime,
        actual_sun_diameter_km: float = 1_392_700,
        min_area: int = 150,
        k_size: tuple = (17, 17),
    ):
        """
        Initializes the Sun class and processes the input solar image.

        Parameters:
            sun_img (Image.Image): Input solar image (RGB).
            date_time (datetime): Date and time for distance calculation.
            actual_sun_diameter_km (float): True solar diameter in kilometers.
            min_area (int): Minimum area in pixels to consider a sunspot.
            k_size (tuple): Gaussian blur kernel size.
        """
        self.sun_img = sun_img
        self.k_size = k_size
        self.min_area = min_area
        self.actual_sun_diameter_km = actual_sun_diameter_km

        self.sun_diameter_pixels = None
        self.sunspots = []
        self.earth_sun_distance_km = None
        self.annotated_img = None

        self.sunspot_mask = self._get_sunspot_mask()
        self.sun_diameter_pixels = self._identify_sun_surface()
        self.earth_sun_distance_km = self._calculate_earth_sun_distance(date_time)
        self.sunspots = self._identify_sunspots()
        self.annotated_img = self._annotate_sunspots()

    def _preprocess_image(self) -> np.ndarray:
        """Converts image to grayscale and applies Gaussian blur."""
        gray = cv2.cvtColor(np.array(self.sun_img), cv2.COLOR_RGB2GRAY)
        return cv2.GaussianBlur(gray, self.k_size, 0)

    def _get_sunspot_mask(self) -> np.ndarray:
        """Creates a binary mask of the sunspots based on binary thresholding."""
        blurred = self._preprocess_image()
        cut_value = self._find_hist_cut_value(blurred.ravel())
        _, mask = cv2.threshold(blurred, cut_value, 255, cv2.THRESH_BINARY)
        return mask

    def _identify_sun_surface(self) -> int:
        """Finds the largest contour assumed to be the sun's disk and returns its diameter in pixels."""
        contours, _ = cv2.findContours(
            self.sunspot_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            raise ValueError("No contours found in sunspot mask.")
        max_contour = max(contours, key=cv2.contourArea)
        _, _, w, h = cv2.boundingRect(max_contour)
        return max(w, h)

    def _find_hist_cut_value(self, pixel_data: np.ndarray) -> int:
        """
        Determines the intensity cut value by detecting plateaus in the histogram.

        Parameters:
            pixel_data (np.ndarray): Flattened grayscale image data.

        Returns:
            int: Cutoff intensity value for thresholding.
        """
        histogram, _ = np.histogram(pixel_data, bins=256, range=(0, 255))
        half_index = len(histogram) // 2
        peak_index = np.argmax(histogram[half_index:]) + half_index

        for i in range(peak_index, 4, -1):
            left_val = histogram[i - 5]
            right_val = histogram[i]
            variation = abs(right_val - left_val) / max(right_val, 1)
            if variation < 0.05:
                return i
        return peak_index

    def _calculate_earth_sun_distance(self, date_time: datetime) -> float:
        """
        Calculates the Earth-Sun distance based on the date.

        Parameters:
            date_time (datetime): The date of observation.

        Returns:
            float: Distance in kilometers.
        """
        mean_distance = 149_597_870.7
        eccentricity = 0.0167
        day_of_year = date_time.timetuple().tm_yday
        angle = 2 * pi * (day_of_year / 365.25)
        return mean_distance * (1 - eccentricity * cos(angle))

    def _identify_sunspots(self) -> List[Dict]:
        """
        Identifies valid sunspots based on contour size filtering.

        Returns:
            List[Dict]: List of sunspot dictionaries with bounding box and pixel diameter.
        """
        if self.sun_diameter_pixels is None:
            raise ValueError("Sun diameter in pixels is not set.")
        if self.earth_sun_distance_km is None:
            raise ValueError("Earth-Sun distance not calculated.")
        px_to_km = self.actual_sun_diameter_km / self.sun_diameter_pixels

        contours, _ = cv2.findContours(
            self.sunspot_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        sunspots = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                diameter = max(w, h)
                if diameter < self.sun_diameter_pixels * 0.9:
                    diameter_km = diameter * px_to_km
                    angular_size_arcmin = (
                        (diameter_km / self.earth_sun_distance_km) * (180 / pi) * 60
                    )
                    sunspots.append(
                        {
                            "bbox": (x, y, w, h),
                            "diameter_pixels": diameter,
                            "diameter_km": diameter_km,
                            "angular_size_arcmin": angular_size_arcmin,
                        }
                    )

        sunspots.sort(key=lambda s: s["diameter_km"], reverse=True)

        for idx, spot in enumerate(sunspots):
            spot["id"] = f"SS{idx + 1:03d}"
        return sunspots

    def _annotate_sunspots(self) -> np.ndarray:
        """
        Draws bounding boxes and labels for sunspots on the original image.

        Returns:
            np.ndarray: Annotated RGB image.
        """
        img = np.array(self.sun_img)
        for spot in self.sunspots:
            x, y, w, h = spot["bbox"]
            label = spot["id"]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 102, 204), 2)
            cv2.putText(
                img,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 102, 204),
                2,
            )
        return img


def download_sun_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        print("Failed to download image from URL:", url)
        return None


def print_sunspot_table(sunspots: list):
    """
    Prints sunspot data in a formatted table.

    Parameters:
        sunspots (list): List of dictionaries containing sunspot data.
    """
    if not sunspots:
        print("No sunspots found.")
        return

    # Format sunspot data for table display
    table_data = []
    for spot in sunspots:
        row = {
            "ID": spot.get("id", ""),
            "BBox": str(spot.get("bbox", "")),
            "Diam (px)": f'{spot.get("diameter_pixels", 0):.1f}',
            "Diam (km)": f'{spot.get("diameter_km", 0):,.0f}',
            "Ang. Size (arcmin)": f'{spot.get("angular_size_arcmin", 0):.3f}',
        }
        table_data.append(row)

    print(tabulate(table_data, headers="keys", tablefmt="grid"))


# Display information about the sunspot detection in a table
image_url = f"https://jsoc1.stanford.edu/data/hmi/images/2024/08/15/20240815_231500_Ic_flat_4k.jpg"
sun_img = download_sun_image(image_url)
sun = Sun(sun_img=sun_img, date_time=datetime(2024, 8, 15, 23, 15))
print_sunspot_table(sun.sunspots)

# Check sunspot annotation on different images
fig_annotated, ax_annotated = plt.subplots(
    2, 6, figsize=(10, 6), sharex="col", sharey="col"
)

for i, j in enumerate(["02", "04", "06", "08", "10", "12"]):
    # get and process sun HMI example image from SDO
    image_url = f"https://jsoc1.stanford.edu/data/hmi/images/2024/{j}/15/2024{j}15_231500_Ic_flat_4k.jpg"
    sun_img = download_sun_image(image_url)
    sun = Sun(sun_img=sun_img, date_time=datetime(2024, int(j), 15, 23, 15))

    ax_annotated[0][i].imshow(sun.sunspot_mask)
    ax_annotated[0][i].set_title(f"2024-{j}-15")
    ax_annotated[1][i].imshow(sun.annotated_img)

plt.show()
