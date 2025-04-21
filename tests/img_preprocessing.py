import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from io import BytesIO

plt.rcParams["image.cmap"] = "gray"
IMAGE_URL = "https://jsoc1.stanford.edu/data/hmi/images/2025/04/15/20250415_231500_Ic_flat_4k.jpg"


def download_sun_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        return None


# get a sun HMI example image from SDO
sun_img = download_sun_image(IMAGE_URL)

# convert to OpenCV grayscale image
gs_img = cv2.cvtColor(np.array(sun_img), cv2.COLOR_RGB2GRAY)

# apply Gaussian blur to reduce noise
blurred_img = cv2.GaussianBlur(gs_img, (13, 13), 0)

# Increase contrast with histogram equalization
equalized_img = cv2.equalizeHist(gs_img)

equalized_blurred = cv2.GaussianBlur(equalized_img, (13, 13), 0)
blurred_equalized = cv2.equalizeHist(blurred_img)

# display histograms
fig_hist, ax_hist = plt.subplots(3, 5, figsize=(10, 6), sharex="row", sharey="row")

ax_hist[0][0].imshow(gs_img)
ax_hist[0][0].set_title("converted gray-scale")
ax_hist[1][0].hist(gs_img.ravel(), bins=256, range=(0, 255))
ax_hist[1][0].set_yscale("log")
ax_hist[1][0].set_xlabel("pixel intensity")
ax_hist[2][0].imshow(cv2.threshold(gs_img, 162, 255, cv2.THRESH_BINARY)[1])

ax_hist[0][1].imshow(blurred_img)
ax_hist[0][1].set_title("gaussian blur")
ax_hist[1][1].hist(blurred_img.ravel(), bins=256, range=(0, 255))
ax_hist[1][1].set_yscale("log")
ax_hist[1][1].set_xlabel("pixel intensity")
ax_hist[2][1].imshow(cv2.threshold(blurred_img, 182, 255, cv2.THRESH_BINARY)[1])

ax_hist[0][2].imshow(equalized_img)
ax_hist[0][2].set_title("equalized")
ax_hist[1][2].hist(equalized_img.ravel(), bins=256, range=(0, 255))
ax_hist[1][2].set_yscale("log")
ax_hist[1][2].set_xlabel("pixel intensity")
ax_hist[2][2].imshow(cv2.threshold(equalized_img, 12, 255, cv2.THRESH_BINARY)[1])

ax_hist[0][3].imshow(equalized_blurred)
ax_hist[0][3].set_title("equalized blurred")
ax_hist[1][3].hist(equalized_blurred.ravel(), bins=256, range=(0, 255))
ax_hist[1][3].set_yscale("log")
ax_hist[1][3].set_xlabel("pixel intensity")
ax_hist[2][3].imshow(cv2.threshold(equalized_blurred, 25, 255, cv2.THRESH_BINARY)[1])

ax_hist[0][4].imshow(blurred_equalized)
ax_hist[0][4].set_title("blurred equalized")
ax_hist[1][4].hist(blurred_equalized.ravel(), bins=256, range=(0, 255))
ax_hist[1][4].set_yscale("log")
ax_hist[1][4].set_xlabel("pixel intensity")
ax_hist[2][4].imshow(cv2.threshold(blurred_equalized, 10, 255, cv2.THRESH_BINARY)[1])

fig_blur, ax_blur = plt.subplots(3, 6, figsize=(12, 6), sharex="row", sharey="row")

ax_blur[0][0].imshow(gs_img)
ax_blur[0][0].set_title("converted gray-scale")
ax_blur[1][0].hist(gs_img.ravel(), bins=256, range=(0, 255))
ax_blur[1][0].set_yscale("log")
ax_blur[1][0].set_xlabel("pixel intensity")
ax_blur[2][0].imshow(cv2.threshold(gs_img, 162, 255, cv2.THRESH_BINARY)[1])

for i, param in enumerate([(9, 178), (13, 182), (17, 184), (21, 186), (25, 188)]):
    blurred = cv2.GaussianBlur(gs_img, (param[0], param[0]), 0)
    ax_blur[0][(i + 1)].imshow(blurred)
    ax_blur[0][(i + 1)].set_title(f"blurred kernel {param[0]}")
    ax_blur[1][(i + 1)].hist(blurred.ravel(), bins=256, range=(0, 255))
    ax_blur[1][(i + 1)].set_yscale("log")
    ax_blur[1][(i + 1)].set_xlabel("pixel intensity")
    ax_blur[2][(i + 1)].imshow(
        cv2.threshold(blurred, param[1], 255, cv2.THRESH_BINARY)[1]
    )


plt.show()
