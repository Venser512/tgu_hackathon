import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
    )

    bot_token: str

    ROOT_PATH: str = os.path.dirname(os.path.abspath(__file__))
    YOLO_PATH: str = ROOT_PATH + "/resources/yolo"
    YOLO_CLASSES_DETECT: list = ['person', 'car', 'bus', "dog","cat"]


settings = Settings()
