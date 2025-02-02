call conda activate mmd
call uvicorn --reload detect_api:app
call conda deactivate mmd
pause