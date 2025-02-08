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

# Определение устройства для вычислений: CUDA, если доступно, иначе CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Загрузка предобученной модели и процессора для изображений
model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

# Определение функции косинусного сходства
cos = torch.nn.CosineSimilarity(dim=1)

# Загрузка данных из JSON-файлов
with open('train_attribution_geo_full_last5.json', encoding="utf-8") as f:
    data_landmark = json.load(f)

with open('museum_description_last3.json', encoding="utf-8") as f1:
    data_museum = json.load(f1)

# Инициализация тензоров для хранения векторов изображений
x = torch.zeros(len(data_museum), 768)
y = torch.zeros(len(data_landmark), 768)

# Заполнение тензора x векторами изображений из data_museum
n = 0
for i in data_museum.keys():
    x[n] = torch.Tensor(data_museum[i]['image_vector'])
    n += 1
x = x.to(device)

# Заполнение тензора y векторами изображений из data_landmark
n = 0
for j in data_landmark.keys():
    y[n] = torch.Tensor(data_landmark[j]['image_vector'])
    n += 1
y = y.to(device)
print(y.shape)

# Загрузка предобученной модели для текста
model2 = SentenceTransformer('DiTy/bi-encoder-russian-msmarco')

# Перемещение моделей на выбранное устройство
model.to(device)
model2.to(device)

def process_image(img_str):
    """
    Обработка изображения, закодированного в base64, для поиска соответствующей достопримечательности.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Очистка кэша CUDA

    # Декодирование изображения из base64
    img_data = base64.b64decode(img_str)
    image = Image.open(io.BytesIO(img_data))

    n = 0
    res = ""
    with torch.no_grad():
        # Получение признаков изображения
        inputs = processor(images=image, return_tensors="pt").to(device)
        image_features = model.get_image_features(**inputs).reshape(1, -1)

        # Вычисление косинусного сходства с векторами достопримечательностей
        l = cos(image_features.to(device), y)
        o = torch.argmax(l).item()

        # Поиск наиболее похожей достопримечательности
        for i in data_landmark.keys():
            if o == n:
                res = i
                break
            n += 1

        print(l[o])
        if l[o].item() > 0.8:
            result = data_landmark[res]
        else:
            result = {'title': 'Данная достопримечательность не найдена', 'location': None, 'descr': [""], 'url': "", 'wikipedia': ""}

    return json.dumps({'title': result['title'], 'location': result['location'], 'descr': result['descr'][0], 'url': result['url'], 'wikipedia': result['wikipedia']})

def process_text(text):
    """
    Обработка текста для поиска соответствующей достопримечательности.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Очистка кэша CUDA

    n = 0
    res = ""
    with torch.no_grad():
        # Получение признаков текста
        text_features = model2.encode(text)
        print(text_features.shape)

        # Поиск наиболее похожей достопримечательности по тексту
        for i in data_landmark.keys():
            results = util.semantic_search(text_features, np.array(data_landmark[i]['text_vector'], dtype='float32'))
            o = results[0][0]['score']

            if o > n:
                res = i
                n = o

        print(n)
        result = data_landmark[res]

        # Кодирование изображения в base64
        finded_image = Image.open(result['filename'])
        buffered = io.BytesIO()
        finded_image.save(buffered, format="JPEG")
        finded_img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return json.dumps({'title': result['title'], 'location': result['location'], 'descr': result['descr'], 'url': result['url'], 'wikipedia': result['wikipedia'], 'image': finded_img_str})

def process_museum_image(img_str):
    """
    Обработка изображения, закодированного в base64, для поиска соответствующей картины в музее.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Очистка кэша CUDA

    # Декодирование изображения из base64
    img_data = base64.b64decode(img_str)
    image = Image.open(io.BytesIO(img_data))

    n = 0
    res = ""
    with torch.no_grad():
        # Получение признаков изображения
        inputs = processor(images=image, return_tensors="pt").to(device)
        image_features = model.get_image_features(**inputs).reshape(1, -1)

        # Вычисление косинусного сходства с векторами картин
        l = cos(image_features.to(device), x)
        o = torch.argmax(l).item()

        # Поиск наиболее похожей картины
        for i in data_museum.keys():
            if o == n:
                res = i
                break
            n += 1

        print(l[o])
        if l[o].item() > 0.8:
            result = data_museum[res]
        else:
            result = {'title': 'Данная картина не найдена', 'author': "", 'descr': ""}

    return json.dumps({'title': result['title'], 'author': result['author'], 'descr': result['descr']})

def process_museum_text(text):
    """
    Обработка текста для поиска соответствующей картины в музее.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Очистка кэша CUDA

    n = 0
    res = ""
    with torch.no_grad():
        # Получение признаков текста
        text_features = model2.encode(text)
        print(text_features.shape)

        # Поиск наиболее похожей картины по тексту
        for i in data_museum.keys():
            results = util.semantic_search(text_features, np.array(data_museum[i]['text_vector'], dtype='float32'))
            o = results[0][0]['score']

            if o > n:
                res = i
                n = o

        print(n)
        result = data_museum[res]

        # Кодирование изображения в base64
        finded_image = Image.open(result['filename'])
        buffered = io.BytesIO()
        finded_image.save(buffered, format="JPEG")
        finded_img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')