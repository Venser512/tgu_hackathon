# -*- coding: cp1251 -*-

from inference import process_image
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

@app.post("/landmark_image/")
async def root (request: Request):
    try:


        contents = await request.body()

        data = json.loads(contents)

        result = process_image(img_str = data['input_image'])

        content = {"Result": result}
        headers = {"Content-Language": "en-US"}

        return JSONResponse(content=content, headers=headers)
    
    except Exception as e:
        logger.error("An error occurred", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")
    

    
    
"""
@app.post("/landmark_text/")
async def root2 (request: Request):

    contents = await request.body()

    data = json.loads(contents)

    result = process_text(image = data['input_text'])

    content = {"Result": result}
    headers = {"Content-Language": "en-US"}
    return JSONResponse(content=content, headers=headers)


@app.post("/landmark_geo/")
async def root3 (request: Request):

    contents = await request.body()

    data = json.loads(contents)

    result = process_geo(lat = data['input_lat'], lon = data['input_lon'])

    content = {"Result": result}
    headers = {"Content-Language": "en-US"}
    return JSONResponse(content=content, headers=headers)
    
    
@app.post("/museum_picture/")
async def root4 (request: Request):

    contents = await request.body()

    data = json.loads(contents)

    result = process_museum_picture(lat = data['input_image'])

    content = {"Result": result}
    headers = {"Content-Language": "en-US"}
    return JSONResponse(content=content, headers=headers)


"""
