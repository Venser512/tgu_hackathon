import json
import os

import cv2
from aiogram import F, Router, types
from aiogram.client import bot
from aiogram.enums import ParseMode
from aiogram.filters import Command

from alg.yolo import apply_yolo_object_detection_by_url

router = Router(name=__name__)

CURRENT_MENU = ''

url = "http://localhost:8000/landmark_image/"
url2 = "http://localhost:8000/museum_image/"
import io
import base64
from PIL import Image
import requests




@router.message(Command("yolo"))
async def yolo_command(message: types.Message):
    global CURRENT_MENU
    CURRENT_MENU = 'yolo'
    await message.answer(
        text="Выбран пункт меню - <b>Поиск объектов по фото (YOLO)</b>. Загрузите фото"
    )


@router.message(Command("sign"))
async def sign_command(message: types.Message):
    global CURRENT_MENU

    CURRENT_MENU = 'sign'
    await message.answer(
        text="Выбран пункт меню - <b>Поиск достопримечательности по фото</b>. Загрузите фото"
    )


@router.message(Command("paint"))
async def paint_command(message: types.Message):
    global CURRENT_MENU

    CURRENT_MENU = 'paint'
    await message.answer(
        text="Выбран пункт меню - <b>Поиск картины по фото</b>. Загрузите фото"
    )


@router.message(F.photo)
async def image_message(message: types.Message):
    global CURRENT_MENU
    path = f'./resources/temp/{message.from_user.id}'

    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
        print("The new directory is created!")

    print(f'CURRENT_MENU: {CURRENT_MENU}')

    file_name = f'{message.photo[-1].file_id}'
    file_dist = f'{path}/{file_name}.jpg'
    print(f'file_dist: {file_dist}')
    await message.bot.download(message.photo[-1], destination=file_dist)

    if CURRENT_MENU == 'yolo':
        file_result, stat = apply_yolo_object_detection_by_url(file_dist)
        cv2.imwrite(f'{path}/{file_name}_result.jpg', file_result)

        if len(stat) > 0:
            mess_text = "Найдено объектов:\n"
            for key, value in stat.items():
                mess_text += f" <b>{key}</b>: {value}\n"
        else:
            mess_text = "Не найдено ни одного объекта!"

        await message.reply_photo(photo=types.FSInputFile(f'{path}/{file_name}_result.jpg'),
                                  caption=mess_text,
                                  parse_mode=ParseMode.HTML)
    elif CURRENT_MENU == 'sign':

        image = Image.open(file_dist)

        buffered = io.BytesIO()
        image.save(buffered, format="BMP")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        with  requests.Session() as client:
           resp_rep = client.post(url, data=json.dumps({"input_image" : img_str}))


        result = json.loads(json.loads(resp_rep.text)['Result'])
        
        await message.answer(text=str(result['title']) + ' ' + str(result['descr']))
    elif CURRENT_MENU == 'paint':

        image = Image.open(file_dist)

        buffered = io.BytesIO()
        image.save(buffered, format="BMP")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        with  requests.Session() as client:
           resp_rep = client.post(url2, data=json.dumps({"input_image" : img_str}))


        result = json.loads(json.loads(resp_rep.text)['Result'])
        
        await message.answer(text=str(result['title']) + ' ' + str(result['descr']))

    else:
        await message.answer(text="Выберите пункт меню перед загрузкой фото")
