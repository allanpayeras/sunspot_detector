import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from io import BytesIO

plt.rcParams["image.cmap"] = "gray"


def download_sun_image(url):
    """
    Downloads an image from the specified URL.
    """
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        print("Failed to download image from URL:", url)
        return None


def preprocess_image(color_image, k_size=(15, 15), sigma=0):
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


fig_hist, ax_hist = plt.subplots(3, 6, figsize=(10, 6))

for i, j in enumerate(["02", "04", "06", "08", "10", "12"]):
    # get a sun HMI example image from SDO
    image_url = f"https://jsoc1.stanford.edu/data/hmi/images/2024/{j}/15/2024{j}15_231500_Ic_flat_4k.jpg"
    sun_img = download_sun_image(image_url)
    blurred_img = preprocess_image(sun_img)

    # Find the histogram cut value
    cut_value = find_cut_value(blurred_img.ravel())
    print(f"Cut value: {cut_value}")

    ax_hist[0][i].imshow(blurred_img)
    ax_hist[1][i].hist(blurred_img.ravel(), bins=256, range=(0, 255))
    ax_hist[1][i].set_yscale("log")
    ax_hist[1][i].set_xlabel("pixel intensity")
    ax_hist[1][i].axvline(
        cut_value, color="red", linestyle="--", label=f"Cut Value: {cut_value}"
    )
    ax_hist[1][i].legend()
    ax_hist[2][i].imshow(
        cv2.threshold(blurred_img, cut_value, 255, cv2.THRESH_BINARY)[1]
    )

plt.show()
