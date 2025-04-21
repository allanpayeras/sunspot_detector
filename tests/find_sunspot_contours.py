import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from io import BytesIO

plt.rcParams["image.cmap"] = "gray"


def download_sun_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        print("Failed to download image from URL:", url)
        return None


def preprocess_image(color_image, k_size=(17, 17), sigma=0):
    """
    Preprocesses the input color image by converting it to grayscale and applying Gaussian blur.

    Parameters:
        color_image (PIL.Image): The input color image.
        ksize (tuple): Kernel size for Gaussian blur.
        sigma (float): Standard deviation for Gaussian blur.

    Returns:
        np.ndarray: The preprocessed grayscale image.
    """
    # Convert to OpenCV grayscale image
    gray_image = cv2.cvtColor(np.array(color_image), cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to reduce granularity
    return cv2.GaussianBlur(gray_image, k_size, sigma)


def find_cut_value(pixel_data):
    """
    Finds the pixel intensity cut value based on the provided algorithm.

    Parameters:
        histogram (np.ndarray): One-dimensional array of pixel intensity values.

    Returns:
        int: The pixel intensity cut value.
    """
    # Step 1: Identify the peak value in the second half of the histogram
    histogram = np.histogram(pixel_data, bins=256, range=(0, 255))[0]
    half_index = len(histogram) // 2
    second_half = histogram[half_index:]
    peak_index = np.argmax(second_half) + half_index

    # Step 2: Monitor the variation going left from the peak
    for i in range(peak_index, 4, -1):  # Start from the peak and move left
        left_value = histogram[i - 5]
        right_value = histogram[i]

        # Calculate the variation percentage
        variation = abs(right_value - left_value) / max(
            right_value, 1
        )  # Avoid division by zero

        # Step 3: Check if the variation is smaller than 5%
        if variation < 0.05:
            return i  # Return the right-side value when the condition is met

    # If no cut value is found, return the peak index as a fallback
    return peak_index


def get_sunspot_mask(sun_img, k_size=(17, 17)):
    """
    Processes the sun image to create a mask for sunspots.

    Parameters:
        sun_img (PIL.Image): The input sun image.
        k_size (tuple): Kernel size for Gaussian blur.

    Returns:
        np.ndarray: The binary mask of sunspots.
    """
    blurred_img = preprocess_image(sun_img, k_size=k_size)
    cut_value = find_cut_value(blurred_img.ravel())
    return cv2.threshold(blurred_img, cut_value, 255, cv2.THRESH_BINARY)[1]


def identify_sunspots(sunspot_mask, min_area=150):
    """
    Identifies sunspots in the binary mask and returns their bounding boxes.

    Parameters:
        sunspot_mask (np.ndarray): The binary mask of sunspots.
        min_area (int): Minimum area to consider a contour as a sunspot.

    Returns:
        list: List of dictionaries with bounding box coordinates for each sunspot.
    """
    contours, _ = cv2.findContours(
        sunspot_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    sunspots = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:  # Filter very small regions
            x, y, w, h = cv2.boundingRect(cnt)
            sunspots.append({"bbox": (x, y, w, h)})
    return sunspots


def drow_sunspots_on_image(sun_img, sunspots):
    """
    Draws bounding boxes around identified sunspots on the original image.

    Parameters:
        sun_img (PIL.Image): The original sun image.
        sunspots (list): List of dictionaries with bounding box coordinates for each sunspot.

    Returns:
        np.ndarray: The image with drawn bounding boxes.
    """
    img_draw = np.array(sun_img)
    for spot in sunspots:
        x, y, w, h = spot["bbox"]
        cv2.rectangle(
            img_draw, (x, y), (x + w, y + h), color=(0, 102, 204), thickness=2
        )
    return img_draw


# Study the effect of area cut on sunspot detection
fig_area_cut, ax_area_cut = plt.subplots(
    2, 6, figsize=(10, 6), sharex=True, sharey=True
)

for i, area_cut in enumerate([50, 100, 150, 200, 250, 300]):
    # get and process sun HMI example image from SDO
    image_url = "https://jsoc1.stanford.edu/data/hmi/images/2024/06/15/20240615_231500_Ic_flat_4k.jpg"
    sun_img = download_sun_image(image_url)
    sunspot_mask = get_sunspot_mask(sun_img)

    # identify and measure sunspots
    sunspots = identify_sunspots(sunspot_mask, min_area=area_cut)

    # draw sunspots on the original image
    img_draw = drow_sunspots_on_image(sun_img.copy(), sunspots)

    ax_area_cut[0][i].imshow(sunspot_mask)
    ax_area_cut[0][i].set_title(f"Area cut: {area_cut}")
    ax_area_cut[1][i].imshow(img_draw)


# Check sunspot detection on different images
fig_id_sunspot, ax_id_sunspot = plt.subplots(
    2, 6, figsize=(10, 6), sharex="col", sharey="col"
)

for i, j in enumerate(["02", "04", "06", "08", "10", "12"]):
    # get and process sun HMI example image from SDO
    image_url = f"https://jsoc1.stanford.edu/data/hmi/images/2024/{j}/15/2024{j}15_231500_Ic_flat_4k.jpg"
    sun_img = download_sun_image(image_url)
    sunspot_mask = get_sunspot_mask(sun_img)

    # identify and measure sunspots
    sunspots = identify_sunspots(sunspot_mask, min_area=150)

    # draw sunspots on the original image
    img_draw = drow_sunspots_on_image(sun_img.copy(), sunspots)

    ax_id_sunspot[0][i].imshow(sunspot_mask)
    ax_id_sunspot[0][i].set_title(f"2024-{j}-15")
    ax_id_sunspot[1][i].imshow(img_draw)

plt.show()
