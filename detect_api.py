# -*- coding: cp1251 -*-

from inference_cpu import process_image, process_text, process_museum_image, process_museum_text
import json
from typing import Any, Dict, AnyStr, List, Union
from collections import OrderedDict
from fastapi import Depends, FastAPI, HTTPException, status, Request, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


@app.post("/test/landmark_image/")
async def root_test(request: Request):
    try:
        content = {
            "Result": {
                "descr": [
                    "Наименование достопримечательности: СокольникиСокольники - это один из самых больших парков в Москве, расположенный на северо-востоке города. Парк занимает площадь около 100 гектаров и является популярным местом отдыха для москвичей и гостей столицы. В парке есть множество достопримечательностей:1. **Лыжная трасса**: На территории парка находится лыжная трасса, которая используется для проведения соревнований по лыжным гонкам и биатлону. В зимний период здесь часто проходят различные спортивные мероприятия.2. **Парковые зоны**: Парк разделен на несколько зон, каждая из которых имеет свои уникальные особенности. Например, в одной из зон можно найти исторический павильон, который был построен в 1954 году и является одним из символов парка.3. **Рекреационные зоны**: В парке есть множество зон для отдыха и развлечений, где можно провести время с семьей или друзьями. Здесь есть детские площадки, места для пикников, а также зоны для занятий спортом.4. **Историческая часть парка**: В центральной части парка находится историческая часть, где можно увидеть старинные здания и памятники архитектуры.5. **Рекреационные зоны для активного отдыха**: В парке есть зоны для занятий спортом и активного отдыха, такие как футбольные поля, беговые дорожки и зоны для занятий фитнесом.Парк Сокольники является важной частью культурного и рекреационного наследия Москвы и привлекает множество посетителей круглый год."
                ],
                "filename": "d:\\gl_train\\0\\0\\0\\00001d45edef6c9e.jpg",
                "location": {
                    "lat": 55.795628,
                    "lon": 37.675195
                },
                "picture": "00001d45edef6c9e",
                "title": "Сокольники",
                "url": "http://commons.wikimedia.org/wiki/File:Sokolniki_District,_Moscow,_Russia_-_panoramio_(67).jpg",
                "wikipedia": ""
            }}

        headers = {"Content-Language": "en-US"}
        print(f"content={json.dumps(content)}")
        return JSONResponse(content=content, headers=headers)


    except Exception as e:
        print(f"Error: {e}")
        logger.error("An error occurred", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/test/museum_image/")
async def root3_test(request: Request):
    content = {
        "Result": {
            "author": "Илья Ефимович Репин",
            "descr": "«Пейзаж Здравнёво» — картина, написанная Ильёй Ефимовичем Репиным в период его пребывания в имении Здравнёво в Белоруссии в конце XIX века. Это произведение отражает жанр пейзажа, который стал популярным в российской живописи наряду с портретами и историческими полотнами. Репин, известный прежде всего как мастер портрета и жанровой картины, здесь демонстрирует своё мастерство в изображении природы.Картина выполнена в реалистическом стиле, характерном для творчества Репина, который стремился к точной передаче видимого мира и эмоционального состояния. В «Пейзаже Здравнёво» наблюдается тонкое внимание к деталям и игра светотени, что позволяет передать атмосферу летнего дня на фоне живописной природы. Репин с любовью изображает поля и леса, создавая гармоничное сочетание цветов и форм.В историческом контексте эта работа отражает стремление художников того времени исследовать и запечатлеть простые, знакомые каждому картины природы, и одновременно искать в них душевное равновесие и вдохновение.",
            "filename": "./data/100549316.jpg",
            "title": "Пейзаж Здравнёво"
        }}

    headers = {"Content-Language": "en-US"}
    return JSONResponse(content=content, headers=headers)


@app.post("/landmark_image/")
async def root(request: Request):
    try:

        contents = await request.body()

        data = json.loads(contents)

        result = process_image(img_str=data['input_image'])

        content = {"Result": result}
        headers = {"Content-Language": "en-US"}

        return JSONResponse(content=content, headers=headers)

    except Exception as e:
        print(f"Error: {e}")
        logger.error("An error occurred", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/landmark_text/")
async def root2(request: Request):
    contents = await request.body()

    data = json.loads(contents)

    result = process_text(text=data['input_text'])

    content = {"Result": result}
    headers = {"Content-Language": "en-US"}
    return JSONResponse(content=content, headers=headers)


@app.post("/museum_image/")
async def root3(request: Request):
    contents = await request.body()

    data = json.loads(contents)

    result = process_museum_image(img_str=data['input_image'])

    content = {"Result": result}
    headers = {"Content-Language": "en-US"}
    return JSONResponse(content=content, headers=headers)


@app.post("/museum_text/")
async def root4(request: Request):
    contents = await request.body()

    data = json.loads(contents)

    result = process_museum_text(text=data['input_text'])

    content = {"Result": result}
    headers = {"Content-Language": "en-US"}
    return JSONResponse(content=content, headers=headers)
