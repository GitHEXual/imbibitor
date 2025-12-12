from aiogram import Router
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton

router = Router()

start_keyboard = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="Старт")]],
    resize_keyboard=True
)


@router.message()
async def handle_message(message: Message):
    if message.text == "Старт":
        await message.answer("Работа начата!", reply_markup=start_keyboard)
    else:
        await message.answer("Нажмите кнопку 'Старт' для начала работы.", reply_markup=start_keyboard)
