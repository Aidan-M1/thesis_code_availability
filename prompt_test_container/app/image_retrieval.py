"""
image_retrieval.py

Retrieves images from dataset for the app to run.

Author: Aidan Murray
Date: 2025-09-26
"""

import pandas as pd
from PIL import Image
import requests
from io import BytesIO
from pathlib import Path

N = 302
# RESOLUTION = (128, 128)
RESOLUTION = (640, 480)
FOLDER_PATH = Path("./images")
FOLDER_PATH.mkdir(exist_ok=True)
DATABASE_PATH = "./data/validation.csv"

# get the image ids and urls without redundency, and sample a random n-values
data = pd.read_csv(DATABASE_PATH)
data = data = data.dropna(subset=['label.name'])
data = data[['point.media.id', 'point.media.path_best']].drop_duplicates()
sample = data.sample(n=N, random_state=42)
image_urls = {k:v for k, v in zip(sample['point.media.id'], sample['point.media.path_best'])}


# download the images at desired resolution
skipped = []
no_iters = 0

for id, url in image_urls.items():
    no_iters += 1
    print(f"Downloading image {no_iters}")

    path = FOLDER_PATH / (str(id) + ".jpg")
    
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to retrieve the image.")
        skipped.append(id)
        continue

    img = Image.open(BytesIO(response.content))
    img_resized = img.resize(RESOLUTION)
    img_resized.save(path)

print(f"Finished. Successfully downloaded {no_iters - len(skipped)} images")