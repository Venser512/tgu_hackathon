__all__ = ("router",)

from aiogram import Router

from .commands import router as commands_router
#from .media_handlers import router as media_router

router = Router(name=__name__)

router.include_routers(
    commands_router,
    #media_router
)
