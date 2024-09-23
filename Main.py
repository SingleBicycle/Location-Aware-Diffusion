import os
import subprocess
import pyarrow.parquet as pq
import json

from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

#subprocess.run(["git", "clone", "https://huggingface.co/datasets/kakaobrain/coyo-700m"])
filename = os.getcwd() +  '\.venv\dataset\metadata.json'
with open(filename, "r") as file:
    data = json.load(file)

url = data[1000][('url')]

#get target image from url
def get_target(target_url):
    response = requests.get(target_url)
    target_image = Image.open(BytesIO(response.content)).convert("RGB")
    return target_image

#display image
#use matplotlib.use('TkAgg',force=True) if get 'Backend tkagg is interactive backend. Turning interactive mode on' error
def show_img(data,id):
    url = data[id]['url']
    print(url)
    img = get_target(url)
    plt.imshow(img)
    return



