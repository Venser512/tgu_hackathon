call tgu\Scripts\activate
call uvicorn --reload detect_api:app
call tgu\Scripts\deactivate
pause