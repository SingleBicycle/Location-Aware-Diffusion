import os
import subprocess
import pyarrow.parquet as pq
import json
import requests
import torch

from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

subprocess.run(["git", "clone", "https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K"])

filename = '/content/LLaVA-CC3M-Pretrain-595K/metadata.json'
with open(filename, "r") as file:
  data = json.load(file)

def get_image_RGB(url):
  response = requests.get(url)
  img = Image.open(BytesIO(response.content)).convert('RGB')
  return img

def show_image(data, id):
  url = data[id]['url']
  print(url)
  img = get_image_RGB(url)
  plt.imshow(img)
  return 

target_texts = [
    ["a picture of a cat", "a photo of a car", "a picture of a mountain", "a photo of an airplane"],
    ["a picture of a laptop", "a photo of a tree", "a picture of a bicycle", "a photo of a person"],
    ["a photo of a cup", "a picture of a dog", "a photo of a bird", "a picture of a bridge"],
    ["a photo of a chair", "a picture of a building", "a photo of a beach", "a picture of a pizza"],
    ["a photo of a clock", "a picture of a suitcase", "a photo of a train", "a picture of a river"],
    ["a picture of a bus", "a photo of a flower", "a picture of a horse", "a photo of a boat"],
    ["a photo of a phone", "a picture of a keyboard", "a photo of a street", "a picture of a bench"],
    ["a photo of a fridge", "a picture of a plate", "a photo of a tree", "a picture of a guitar"],
    ["a picture of a cloud", "a photo of a bicycle", "a picture of a bed", "a photo of a ball"],
    ["a photo of a television", "a picture of a plane", "a photo of a computer", "a picture of a shoe"]
]

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

def detect_object(processor, model, url, texts, threshold):
    image = get_image_RGB(url)
    inputs = processor(text = texts, images = image, return_tensors="pt")
    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])

    #input threshold: the score probability to filter. If score < threshold, an object is not detected.
    # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
    # result contains scores: the probability the object present, labels: the label of the object e.g house is 0. dog is 1 etc, based on your input text, boxes: bouding boxes.
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=threshold)

    return results

import csv

def save_results_to_csv(results, image_id, image_url, file_name="output.csv", write_header=False):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header once, if it's the first run
        if write_header:
            writer.writerow(["Image ID", "Image URL", "Label", "Score", "Box (xmin, ymin, xmax, ymax)"])
        
        # Write each detection result to the CSV file
        for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
            # Flatten the box (bounding box coordinates) to a string or list of floats
            box = [round(coord.item(), 2) for coord in box]  # Rounds each box coordinate
            label = target_texts[label]  # Get the label description from `target_texts`
            writer.writerow([image_id, image_url, label, round(score.item(), 3), box])

data_3000 = data[:3000]
url_3000 = []
for i in range(3000):
  url_3000.append(data_3000[i]['url'])

for i in range(100):
  for j in range(len(target_texts)):
    try:
      result_holder = detect_object(processor, model, url_3000[i], target_texts[j], 0.5)
    except:
      print("not a valid image format")
      continue
    save_results_to_csv(result_holder[0], data[i]['id'], url_3000[i], "output.csv")
  print("finihsed")
    #boxes, scores, labels = result_holder[0]["boxes"], result_holder[0]["scores"], result_holder[0]["labels"]
    #if len(scores) == 0:
      #print("No result detected for this image")
      #continue
    #for box, score, label in zip(boxes, scores, labels):
      #box = [round(i, 2) for i in box.tolist()]
      #print(f"Detected {target_texts[label]} with confidence {round(score.item(), 3)} at location {box}")