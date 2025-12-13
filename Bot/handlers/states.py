from aiogram.fsm.state import State, StatesGroup


class MenuStates(StatesGroup):
    """Состояния меню бота"""
    start = State()  # Стартовое меню
    main = State()  # Главное меню
    chat = State()  # Режим чата с LLM
    settings = State()  # Меню настроек
    select_model = State()  # Выбор модели

