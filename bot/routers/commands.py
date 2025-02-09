import json
import os

import io
import base64

import cv2
import requests
from PIL import Image
from aiogram import F, Router, types
from aiogram.client import bot
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import FSInputFile

from config import settings

router = Router(name=__name__)

CURRENT_MENU = ''


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


@router.message(Command("sign_text"))
async def sign_command(message: types.Message):
    global CURRENT_MENU

    CURRENT_MENU = 'sign_text'
    await message.answer(
        text="Выбран пункт меню - <b>Поиск достопримечательности по тексту</b>. Введите название достопримечательности"
    )


@router.message(Command("paint_text"))
async def paint_command(message: types.Message):
    global CURRENT_MENU

    CURRENT_MENU = 'paint_text'
    await message.answer(
        text="Выбран пункт меню - <b>Поиск картины по названию</b>. Введите название картины"
    )


@router.message(F.photo)
async def image_message(message: types.Message):
    global CURRENT_MENU
    path = f'./resources/temp/{message.from_user.id}'

    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    file_name = f'{message.photo[-1].file_id}'
    file_dist = f'{path}/{file_name}.jpg'
    await message.bot.download(message.photo[-1], destination=file_dist)

    if CURRENT_MENU == 'sign':
        try:
            url = settings.API_URL + "/landmark_image/"

            image = Image.open(file_dist)

            buffered = io.BytesIO()
            image.save(buffered, format="BMP")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

            with requests.Session() as client:
                resp_rep = client.post(url, data=json.dumps({"input_image": img_str}))

            if resp_rep.status_code == 200:
                result = json.loads(json.loads(resp_rep.text)['Result'])
                if str(result['descr']) != '':
                    await message.answer(text=str(f"""
                            <b>{str(result['title'])}</b>\n{str(result['url'])}\n\n{str(result['descr'])} 
                            """), parse_mode=ParseMode.HTML)
                else:
                    await message.answer(text=str(f"<i>{result['title']}</i>"), parse_mode=ParseMode.HTML)
        except Exception as e:
            print(f"Error: {str(e)}")
            await message.answer(text=f"Произошла ошибка: [{e}]")
    elif CURRENT_MENU == 'paint':
        try:
            url = settings.API_URL + "/museum_image/"

            image = Image.open(file_dist)

            buffered = io.BytesIO()
            image.save(buffered, format="BMP")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

            with requests.Session() as client:
                resp_rep = client.post(url, data=json.dumps({"input_image": img_str}))

            if resp_rep.status_code == 200:
                result = json.loads(json.loads(resp_rep.text)['Result'])
                if str(result['descr']) != '':
                    await message.answer(text=str(
                        f"""<b>{str(result['author'])}</b>. {str(result['title'])}\n\n{str(result['descr'])}"""))
                else:
                    await message.answer(text=str(f"<i>{result['title']}</i>"))
        except Exception as e:
            await message.answer(text=f"Произошла ошибка: [{e}]")
    else:
        await message.answer(text="Выберите пункт меню перед загрузкой фото")


@router.message(F.text)
async def paint_text_command(message: types.Message):
    global CURRENT_MENU

    if CURRENT_MENU == 'sign_text':
        try:
            url = settings.API_URL + "/landmark_text/"

            with requests.Session() as client:
                resp_rep = client.post(url, data=json.dumps({"input_text": message.text}))

            if resp_rep.status_code == 200:
                result = json.loads(json.loads(resp_rep.text)['Result'])
                img_data = base64.b64decode(result['image'])

                path = f'./resources/temp/{message.from_user.id}'
                isExist = os.path.exists(path)
                if not isExist:
                    os.makedirs(path)

                imagePath = path + "/" + str(message.message_id) + ".jpg"
                print(f"imagePath: {imagePath}")
                img = Image.open(io.BytesIO(img_data))
                img.save(imagePath, 'jpeg')
                print("saveImage")

                await message.bot.send_photo(chat_id=message.from_user.id,
                                             photo=FSInputFile(path=imagePath))
            else:
                await message.answer(text="Ошибка")
        except Exception as e:
            print(f"Error: {str(e)}")
            await message.answer(text=f"Произошла ошибка: [{e}]")
    elif CURRENT_MENU == 'paint_text':
        try:
            url = settings.API_URL + "/museum_text/"

            with requests.Session() as client:
                resp_rep = client.post(url, data=json.dumps({"input_text": message.text}))

            if resp_rep.status_code == 200:
                try:
                    print(resp_rep.text)
                    result = json.loads(json.loads(resp_rep.text)['Result'])
                    img_data = base64.b64decode(result['image'])

                    path = f'./resources/temp/{message.from_user.id}'
                    isExist = os.path.exists(path)
                    if not isExist:
                        os.makedirs(path)

                    imagePath = path + "/" + str(message.message_id) + ".jpg"
                    print(f"imagePath: {imagePath}")
                    img = Image.open(io.BytesIO(img_data))
                    img.save(imagePath, 'jpeg')
                    print("saveImage")

                    await message.bot.send_photo(chat_id=message.from_user.id,
                                                 photo=FSInputFile(path=imagePath))
                except Exception as e:
                    await message.answer(text=f"Картина не найдена")
            else:
                await message.answer(text="Ошибка")
        except Exception as e:
            print(f"Error: {str(e)}")
            await message.answer(text=f"Произошла ошибка: [{e}]")
    else:
        await message.answer(text="Выберите пункт меню перед вводом текста")
