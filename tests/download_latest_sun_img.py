import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO


def download_latest_sdo_image():
    """
    Downloads the latest SDO HMI 'Ic_flat' image as a 4K JPG from JSOC.
    """
    # URL to fetch the list of image times
    latest_url = "https://jsoc1.stanford.edu/data/hmi/images/image_times"

    # Fetch the list of images
    response = requests.get(latest_url)
    response.raise_for_status()

    # Extract the third line and get the Ic_flat part
    lines = response.text.strip().splitlines()
    if len(lines) < 3:
        raise ValueError("Unexpected file format: fewer than 3 lines found.")

    third_line = lines[2]
    parts = third_line.split()
    if len(parts) < 2:
        raise ValueError("No 'Ic_flat' filename found in the third line.")
    ic_flat_name = parts[1]

    image_url = ic_flat_name + "_4k.jpg"
    print(f"Downloading image from: {image_url}")

    # Download the image
    image_response = requests.get(image_url)
    image_response.raise_for_status()

    return Image.open(BytesIO(image_response.content))


plt.imshow(download_latest_sdo_image())

plt.show()
