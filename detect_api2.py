# -*- coding: cp1251 -*-

from inference2 import process_museum_image
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



@app.post("/museum_image/")
async def root3 (request: Request):

    contents = await request.body()

    data = json.loads(contents)

    
    result = process_museum_image(img_str = data['input_image'])
    

    content = {"Result": result}
    headers = {"Content-Language": "en-US"}
    return JSONResponse(content=content, headers=headers)
    
    




