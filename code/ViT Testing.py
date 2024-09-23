import requests
from PIL import Image
import torch

from transformers import OwlViTProcessor, OwlViTForObjectDetection

#model
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

#Object detection
#text is what class to detect
#Example: texts = [["a photo of a house", "a photo of a dog", "a photo of 3", "a photo of train"]], use chatgpt to generate a list of such objects.
def DetectObject(processor, model, url, texts):
    image = get_image(url)
    inputs = processor(text = texts, images = image, return_tensors="pt")
    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])

    #input threshold: the score probability to filter. If score < threshold, an object is not detected.
    # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
    # result contains scores: the probability the object present, labels: the label of the object e.g house is 0. dog is 1 etc, based on your input text, boxes: bouding boxes.
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.4)

    return results