from PIL import Image
import requests



import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import os
import time
import pickle  # Для сохранения и загрузки признаков

# 1. Загрузка данных
df = pd.read_csv("augmented_description2_extended.csv")

# 2. Определение функции для предобработки новых изображений
def preprocess_new_image(img):
    """
    Загружает, изменяет размер и нормализует одно изображение.
    """
    try:
        #img = load_img(image_path, target_size=(224, 224))
        target_size=(224, 224)
        img = img.resize(target_size)
        img = img_to_array(img) / 255.0  # Нормализация
        img = np.expand_dims(img, axis=0)  # Добавление размерности батча
        return img
    except Exception as e:
        print(f"Ошибка при загрузке изображения: {e}")
        return None


# 3. Загрузка предварительно обученной модели
model_filename = "feature_extraction_model.h5"
try:
    feature_extraction_model = tf.keras.models.load_model(model_filename)
    # Компиляция модели (важно после загрузки, если модель будет использоваться для обучения или оценки)
    feature_extraction_model.compile(optimizer='adam', loss='mse')
    print(f"Модель успешно загружена и скомпилирована из файла: {model_filename}")
except Exception as e:
    print(f"Ошибка при загрузке модели из файла {model_filename}: {e}")
    exit()

# 4. Извлечение и сохранение признаков всего датасета (выполняется только один раз!)
features_filename = "image_features.pkl" # Имя файла для сохранения признаков

print(f"Загрузка признаков всего датасета из файла: {features_filename}")
with open(features_filename, "rb") as f:
       image_features = pickle.load(f)  # Загрузка признаков из файла


def find_closest_image(img, image_features, df, model):
    """
    Находит наиболее похожее изображение в датасете на основе косинусного расстояния между признаками.

    Аргументы:
        image_path: Путь к изображению, для которого нужно найти похожее.
        image_features: Матрица признаков для всех изображений в датасете (предварительно извлеченная).
        df: DataFrame с информацией об изображениях (название, автор, описание).
        model: Загруженная модель для извлечения признаков (не используется, но оставлена для совместимости).

    Возвращает:
        Кортеж: (название, автор, описание) наиболее похожего изображения, или (None, None, None) в случае ошибки.
    """
    try:
        # Предобработка нового изображения
        new_img = preprocess_new_image(img)

        if new_img is None:
            return None, None, None

        # Извлечение признаков для нового изображения
        new_features = model.predict(new_img)

        # Вычисление косинусного сходства
        similarity_scores = cosine_similarity(new_features, image_features) # Вычисляем косинусное сходство

        # Поиск наиболее похожего изображения
        closest_image_index = np.argmax(similarity_scores)
        print(similarity_scores[0][closest_image_index])
        
        if similarity_scores[0][closest_image_index] > 0.95:
          # Получение информации о наиболее похожем изображении
          closest_image_description = df.iloc[closest_image_index]["GPT4DESC"]
          closest_image_title = df.iloc[closest_image_index]["ru_title"]
          closest_image_author = df.iloc[closest_image_index]["ru_author"]
        else:
          closest_image_title = 'Картина не найдена'
          closest_image_author = ''
          closest_image_description = ''

        return closest_image_title, closest_image_author, closest_image_description
    except Exception as e:
        print(f"Ошибка при поиске ближайшего изображения: {e}")
        return None, None, None






def process_museum_image(img_str):
    if torch.cuda.is_available():
        torch.cuda.empty_cache
    img_data = base64.b64decode(img_str)
    image = Image.open(io.BytesIO(img_data))

    
    closest_title, closest_author, closest_description = find_closest_image(image, image_features, df, feature_extraction_model)


    result =  {'title': closest_title, 'author' : closest_author, 'descr' : closest_description}
    return json.dumps({'title': result['title'], 'author' : result['author'],'descr' : result['descr']})



