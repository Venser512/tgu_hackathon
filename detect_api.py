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
                    "������������ ���������������������: �������������������� - ��� ���� �� ����� ������� ������ � ������, ������������� �� ������-������� ������. ���� �������� ������� ����� 100 �������� � �������� ���������� ������ ������ ��� ��������� � ������ �������. � ����� ���� ��������� ����������������������:1. **������ ������**: �� ���������� ����� ��������� ������ ������, ������� ������������ ��� ���������� ������������ �� ������ ������ � ��������. � ������ ������ ����� ����� �������� ��������� ���������� �����������.2. **�������� ����**: ���� �������� �� ��������� ���, ������ �� ������� ����� ���� ���������� �����������. ��������, � ����� �� ��� ����� ����� ������������ ��������, ������� ��� �������� � 1954 ���� � �������� ����� �� �������� �����.3. **������������� ����**: � ����� ���� ��������� ��� ��� ������ � �����������, ��� ����� �������� ����� � ������ ��� ��������. ����� ���� ������� ��������, ����� ��� ��������, � ����� ���� ��� ������� �������.4. **������������ ����� �����**: � ����������� ����� ����� ��������� ������������ �����, ��� ����� ������� ��������� ������ � ��������� �����������.5. **������������� ���� ��� ��������� ������**: � ����� ���� ���� ��� ������� ������� � ��������� ������, ����� ��� ���������� ����, ������� ������� � ���� ��� ������� ��������.���� ���������� �������� ������ ������ ����������� � �������������� �������� ������ � ���������� ��������� ����������� ������� ���."
                ],
                "filename": "d:\\gl_train\\0\\0\\0\\00001d45edef6c9e.jpg",
                "location": {
                    "lat": 55.795628,
                    "lon": 37.675195
                },
                "picture": "00001d45edef6c9e",
                "title": "����������",
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
            "author": "���� �������� �����",
            "descr": "������� �������� � �������, ���������� ����� ���������� ������� � ������ ��� ���������� � ������ �������� � ���������� � ����� XIX ����. ��� ������������ �������� ���� �������, ������� ���� ���������� � ���������� �������� ������ � ���������� � ������������� ���������. �����, ��������� ������ ����� ��� ������ �������� � �������� �������, ����� ������������� ��� ���������� � ����������� �������.������� ��������� � �������������� �����, ����������� ��� ���������� ������, ������� ��������� � ������ �������� �������� ���� � �������������� ���������. � �������� �������� ����������� ������ �������� � ������� � ���� ���������, ��� ��������� �������� ��������� ������� ��� �� ���� ���������� �������. ����� � ������� ���������� ���� � ����, �������� ����������� ��������� ������ � ����.� ������������ ��������� ��� ������ �������� ���������� ���������� ���� ������� ����������� � ����������� �������, �������� ������� ������� �������, � ������������ ������ � ��� �������� ���������� � �����������.",
            "filename": "./data/100549316.jpg",
            "title": "������ ��������"
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
