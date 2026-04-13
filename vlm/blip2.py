import torch
import os
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"

os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
# import requests
# from PIL import Image
# from transformers import Blip2Processor, Blip2ForConditionalGeneration

# processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# # model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map="auto")
# model = Blip2ForConditionalGeneration.from_pretrained(
#     "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
# )
# img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# question = "how many dogs are in the picture?"
# inputs = processor(raw_image, question, return_tensors="pt").to("cuda", torch.float16)

# out = model.generate(**inputs)
# print(out)
# print(processor.decode(out[0], skip_special_tokens=True))
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# model = Blip2ForConditionalGeneration.from_pretrained(
#     "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
# )
# processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map="auto")
# model.to(device)
image_path = "/tmp/capture.jpg"
image = Image.open(image_path).convert("RGB")

question = "This is a scene of"
inputs = processor(images=image, text=question, return_tensors="pt").to(device, torch.float16)
# inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)