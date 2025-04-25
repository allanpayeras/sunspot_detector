import cv2
import requests

import numpy as np

from PIL import Image
from io import BytesIO
from datetime import datetime, timezone
from math import pi, cos
from typing import List, Dict


class Sun:
    def __init__(
        self,
        date_time="latest",
        min_area: int = 150,
        k_size: tuple = (17, 17),
    ):
        """
        Initializes the Sun class and processes the corresponding solar image.

        Parameters:
            date_time (datetime): Date and time to retrieve sun image.
            min_area (int): Minimum area in pixels to consider a sunspot.
            k_size (tuple): Gaussian blur kernel size.
        """
        self.date_time = date_time
        self.k_size = k_size
        self.min_area = min_area
        self.actual_sun_diameter_km = float(1_392_700)

        self.sun_diameter_pixels = None
        self.sunspots = []
        self.earth_sun_distance_km = None
        self.annotated_img = None

        self.sun_img, self.date_time = self._download_image()
        self.sunspot_mask = self._get_sunspot_mask()
        self.sun_diameter_pixels = self._identify_sun_surface()
        self.earth_sun_distance_km = self._calculate_earth_sun_distance()
        self.sunspots = self._identify_sunspots()
        self.annotated_img = self._annotate_sunspots()

    def _download_image(self) -> tuple[Image.Image, datetime]:
        if self.date_time == "latest":
            return self._download_latest_sdo_image()
        else:
            return self._download_sdo_image_by_date(), self.date_time

    def _download_latest_sdo_image(self) -> tuple[Image.Image, datetime]:
        latest_images_url = "https://jsoc1.stanford.edu/data/hmi/images/image_times"
        response = requests.get(latest_images_url)
        response.raise_for_status()

        # Extract date and time of latest image from the first line
        lines = response.text.strip().splitlines()
        if len(lines) < 3:
            raise ValueError(
                f"Unexpected file format (in {latest_images_url}): fewer than 3 lines found."
            )
        first_line = lines[0]
        parts = first_line.split()
        if len(parts) < 2:
            raise ValueError(
                f"No date and time found in the first line of {latest_images_url}."
            )
        date_time_str = parts[1]
        date_time = datetime.strptime(date_time_str, "%Y%m%d_%H%M%S").replace(
            tzinfo=timezone.utc
        )

        # Get the URL of the Ic_flat image from the third line
        third_line = lines[2]
        parts = third_line.split()
        if len(parts) < 2:
            raise ValueError(
                f"No 'Ic_flat' filename found in the third line of {latest_images_url}."
            )
        ic_flat_name = parts[1]
        image_url = ic_flat_name + "_4k.jpg"

        # Download the image
        image_response = requests.get(image_url)
        image_response.raise_for_status()

        return Image.open(BytesIO(image_response.content)), date_time

    def _download_sdo_image_by_date(self) -> Image.Image:
        year = self.date_time.year
        month = self.date_time.month
        day = self.date_time.day
        hour = self.date_time.hour
        minute = self.date_time.minute

        image_url = f"https://jsoc1.stanford.edu/data/hmi/images/{year}/{month:02d}/{day:02d}/{year}{month:02d}{day:02d}_{hour:02d}{minute:02d}00_Ic_flat_4k.jpg"

        response = requests.get(image_url)
        response.raise_for_status()

        return Image.open(BytesIO(response.content))

    def _get_sunspot_mask(self) -> np.ndarray:
        """Creates a binary mask of the sunspots based on binary thresholding."""
        blurred = self._preprocess_image()
        cut_value = self._find_hist_cut_value(blurred.ravel())
        _, mask = cv2.threshold(blurred, cut_value, 255, cv2.THRESH_BINARY)
        return mask

    def _preprocess_image(self) -> np.ndarray:
        """Converts image to grayscale and applies Gaussian blur."""
        gray = cv2.cvtColor(np.array(self.sun_img), cv2.COLOR_RGB2GRAY)
        return cv2.GaussianBlur(gray, self.k_size, 0)

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

    def _calculate_earth_sun_distance(self) -> float:
        """Calculates the Earth-Sun distance based on the date."""
        mean_distance = 149_597_870.7
        eccentricity = 0.0167
        day_of_year = self.date_time.timetuple().tm_yday
        angle = 2 * pi * (day_of_year / 365.25)
        return mean_distance * (1 - eccentricity * cos(angle))

    def _identify_sunspots(self) -> List[Dict]:
        """Identifies valid sunspots based on contour size filtering."""
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
                if diameter < self.sun_diameter_pixels * 0.95:
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
        """Draws bounding boxes and labels for sunspots on the original image."""
        img = np.array(self.sun_img)
        for spot in self.sunspots:
            x, y, w, h = spot["bbox"]
            label = spot["id"]
            cv2.rectangle(img, (x, y), (x + w, y + h), (128, 0, 128), 2)
            cv2.putText(
                img,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (128, 0, 128),
                2,
            )
        return img

    def highlight_sunspots(self, highlight_spot_ids=None) -> np.ndarray:
        """
        Highlight bounding boxes and labels for sunspots on the original image.

        Parameters:
            highlight_spot_id (List): The IDs of the sunspots to highlight. If None, all sunspots are normal.

        Returns:
            np.ndarray: Annotated RGB image.
        """
        img = np.array(self.sun_img)
        for spot in self.sunspots:
            x, y, w, h = spot["bbox"]
            label = spot["id"]
            # Check if this sunspot should be highlighted
            if label in highlight_spot_ids:
                color = (128, 0, 128)
                text_color = (128, 0, 128)
            else:
                color = (192, 192, 192)
                text_color = (192, 192, 192)

            # Draw the rectangle and label
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                img,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                text_color,
                2,
            )
        return img
