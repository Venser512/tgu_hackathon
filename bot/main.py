import asyncio
import logging

from aiogram import Bot
from aiogram import Dispatcher
from aiogram.client.default import DefaultBotProperties

from aiogram.enums import ParseMode
from aiogram.types import BotCommand

from config import settings
from routers import router as main_router


async def set_main_menu(bot: Bot):
    # Создаем список с командами и их описанием для кнопки menu
    main_menu_commands = [
        BotCommand(command='/yolo',
                   description='Поиск объектов на картинке'),
        BotCommand(command='/sign',
                   description='Поиск достопримечательности по фото'),
        BotCommand(command='/paint',
                   description='Поиск картины по фото'),
        BotCommand(command='/help',
                   description='Помощь')
    ]

    await bot.set_my_commands(main_menu_commands)


async def main():
    dp = Dispatcher()
    dp.include_router(main_router)
    dp.startup.register(set_main_menu)

    logging.basicConfig(level=logging.INFO)
    bot = Bot(
        token=settings.bot_token,
        default=DefaultBotProperties(
            parse_mode=ParseMode.HTML,
        )
    )
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
