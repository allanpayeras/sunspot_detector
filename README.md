# 🌞 Visible Sunspot Detector


## 1. Overview

**Visible Sunspot Detector** is a Streamlit-based Python web application that downloads the latest (or user-specified) image of the Sun from NASA's SDO HMI dataset and uses computer vision techniques to detect and analyze sunspots. It estimates their physical dimensions and visibility from Earth and presents this information in an interactive interface.


## 2. Description of the Sunspot Detection Method

The detection process uses OpenCV to identify sunspots from HMI images. The main steps are:

- **Image Preprocessing**: The image is converted to grayscale and blurred with a Gaussian filter to reduce noise.
- **Adaptive Thresholding**: The image histogram is analyzed to determine an appropriate cut-off intensity for segmenting darker regions.
- **Contour Detection**: Binary thresholding followed by `cv2.findContours()` is used to locate closed contours corresponding to sunspots.
- **Filtering**: Very small contours are discarded based on an area cut value.

Each valid sunspot is annotated with a bounding box and assigned an ID such as `SS001`, `SS002`, etc., based on size.


## 3. Estimation of Sunspot Sizes

The app estimates the **real-world diameter** of sunspots using the following approach:

- The diameter of the Sun is known: **1,392,700 km**.
- The image provides the Sun’s diameter in pixels, allowing calculation of a **pixel-to-kilometer conversion factor**.
- For each sunspot, the pixel-based diameter is converted into kilometers.

### Angular Size Calculation

The **angular size** of each sunspot (how large it appears from Earth) is calculated in arcminutes using:

```
angular_size = (diameter_km / earth_sun_distance_km) * (180 / π) * 60
```

The Earth-Sun distance is estimated using the date and Earth’s orbital parameters.


## 4. Criteria for Visibility Classification

Sunspot visibility with the **unaided human eye** depends on its angular size in arcminutes. The following thresholds are used to classify visibility:

| Angular Size (arcmin) | Visibility         |
|------------------------|--------------------|
| > 2.7                  | Very easy          |
| 2.0 – 2.7              | Easy               |
| 1.0 – 2.0              | Possible           |
| 0.75 – 1.0             | Difficult          |
| 0.5 – 0.75             | Very difficult     |
| ≤ 0.5                  | Not visible        |

Visible spots are highlighted in color-coded rows in the Streamlit table.

## 5. Installation and Running the Project

### Requirements

Ensure you have Python 3.8+ installed. Install the required packages with:

```bash
pip install -r requirements.txt
```

### Running the App

To start the Streamlit app, run:

```bash
streamlit run sunspot_app.py
```

Then open the displayed URL in your browser (usually `http://localhost:8501`).

