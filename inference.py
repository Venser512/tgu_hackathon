from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
import json
import io
import base64
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer, util

model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

cos = torch.nn.CosineSimilarity(dim=1)

with open('train_attribution_geo_full_last4.json', encoding="utf-8") as f:
    data_landmark = json.load(f)
    
with open('museum_description_last2.json', encoding="utf-8") as f1:
    data_museum = json.load(f1)



model2 = SentenceTransformer('DiTy/bi-encoder-russian-msmarco')



def process_image(img_str):

    img_data = base64.b64decode(img_str)
    image = Image.open(io.BytesIO(img_data))


    n = 0
    res = ""
    inputs = processor(images=image, return_tensors="pt")
    image_features = model.get_image_features(**inputs)

    for i in data_landmark.keys():
       o = cos(image_features, torch.Tensor(data_landmark[i]['image_vector']))
       if o > n:
          res = i
          n = o
    print(n)      
    result = data_landmark[res]
    finded_image =  Image.open(result['filename'])
    buffered = io.BytesIO()
    finded_image.save(buffered, format="JPEG")
    finded_img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return json.dumps({'title': result['title'], 'location' : result['location'],'descr' : result['descr'], 'url' : result['url'] , 'wikipedia' : result['wikipedia'], 'image' : finded_img_str })



def process_text(text):

    n = 0
    res = ""

    text_features = model2.encode(text)
    print(text_features.shape)
    for i in data_landmark.keys():
       results = util.semantic_search(text_features, np.array(data_landmark[i]['text_vector'],dtype='float32'))
       o = results[0][0]['score']
       
       if o > n:
          res = i
          n = o
    print(n)      
    result = data_landmark[res]
    finded_image =  Image.open(result['filename'])
    buffered = io.BytesIO()
    finded_image.save(buffered, format="JPEG")
    finded_img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return json.dumps({'title': result['title'], 'location' : result['location'],'descr' : result['descr'], 'url' : result['url'] , 'wikipedia' : result['wikipedia'], 'image' : finded_img_str })




def process_museum_image(img_str):

    img_data = base64.b64decode(img_str)
    image = Image.open(io.BytesIO(img_data))


    n = 0
    res = ""
    inputs = processor(images=image, return_tensors="pt")
    image_features = model.get_image_features(**inputs)

    for i in data_museum.keys():
       o = cos(image_features, torch.Tensor(data_museum[i]['image_vector']))
       if o > n:
          res = i
          n = o
    print(n)      
    result = data_museum[res]
    finded_image =  Image.open(result['filename'])
    buffered = io.BytesIO()
    finded_image.save(buffered, format="JPEG")
    finded_img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return json.dumps({'title': result['title'], 'author' : result['author'],'descr' : result['descr'], 'image' : finded_img_str })



def process_museum_text(text):

    n = 0
    res = ""

    text_features = model2.encode(text)
    print(text_features.shape)
    for i in data_museum.keys():
       results = util.semantic_search(text_features, np.array(data_museum[i]['text_vector'],dtype='float32'))
       o = results[0][0]['score']
       
       if o > n:
          res = i
          n = o
    print(n)      
    result = data_museum[res]
    finded_image =  Image.open(result['filename'])
    buffered = io.BytesIO()
    finded_image.save(buffered, format="JPEG")
    finded_img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return json.dumps({'title': result['title'], 'author' : result['author'],'descr' : result['descr'], 'image' : finded_img_str })