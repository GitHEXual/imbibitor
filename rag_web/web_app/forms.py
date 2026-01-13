"""WTForms forms for Flask application."""

from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import StringField, PasswordField, SelectField, SubmitField, BooleanField, IntegerField, FloatField
from wtforms.validators import DataRequired, Email, EqualTo, Length, ValidationError, NumberRange, Optional
import re
from web_app.models import User


class LoginForm(FlaskForm):
    """Login form."""
    
    username = StringField('Имя пользователя', validators=[DataRequired()])
    password = PasswordField('Пароль', validators=[DataRequired()])
    remember_me = BooleanField('Запомнить меня')
    submit = SubmitField('Войти')


class RegisterForm(FlaskForm):
    """Registration form."""
    
    username = StringField(
        'Имя пользователя',
        validators=[
            DataRequired(),
            Length(min=3, max=80, message='Имя пользователя должно быть от 3 до 80 символов')
        ]
    )
    email = StringField('Email (необязательно)', validators=[Email()])
    password = PasswordField(
        'Пароль',
        validators=[
            DataRequired(),
            Length(min=6, message='Пароль должен быть не менее 6 символов')
        ]
    )
    confirm_password = PasswordField(
        'Подтвердите пароль',
        validators=[
            DataRequired(),
            EqualTo('password', message='Пароли не совпадают')
        ]
    )
    submit = SubmitField('Зарегистрироваться')
    
    def validate_username(self, username):
        """Validate unique username."""
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('Это имя пользователя уже занято. Пожалуйста, выберите другое.')
    
    def validate_email(self, email):
        """Validate unique email if provided."""
        if email.data:
            user = User.query.filter_by(email=email.data).first()
            if user:
                raise ValidationError('Этот email уже используется. Пожалуйста, используйте другой.')


def validate_url(form, field):
    """Custom URL validator that accepts localhost and IP addresses."""
    url = field.data
    if not url:
        return
    
    # Simple regex to check for http:// or https:// followed by host:port
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # IP address
        r'[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*)'  # domain
        r'(:\d+)?'  # optional port
        r'(/.*)?$'  # optional path
    )
    
    if not url_pattern.match(url):
        raise ValidationError('Введите корректный URL (например: http://localhost:11434)')


class OllamaSettingsForm(FlaskForm):
    """Ollama settings form."""
    
    ollama_url = StringField(
        'Ollama URL',
        validators=[
            DataRequired(),
            validate_url
        ],
        default='http://localhost:11434'
    )
    embedding_model = SelectField(
        'Модель для embeddings',
        validators=[DataRequired()],
        choices=[],
        coerce=str
    )
    llm_model = SelectField(
        'Модель для генерации текста',
        validators=[DataRequired()],
        choices=[],
        coerce=str
    )
    submit = SubmitField('Сохранить настройки')


class RAGQueryForm(FlaskForm):
    """RAG query form for search and generation."""
    
    query = StringField(
        'Поисковый запрос',
        validators=[DataRequired()],
        render_kw={'placeholder': 'Введите запрос для поиска...'}
    )
    topic = StringField(
        'Тема для генерации поста (необязательно)',
        validators=[Optional()],
        render_kw={'placeholder': 'Оставьте пустым, если нужен только поиск'}
    )
    top_k = IntegerField(
        'Количество результатов',
        validators=[DataRequired(), NumberRange(min=1, max=50)],
        default=5
    )
    use_hybrid = BooleanField(
        'Использовать гибридный поиск (семантический + keyword)',
        default=True
    )
    semantic_weight = FloatField(
        'Вес семантического поиска',
        validators=[DataRequired(), NumberRange(min=0.0, max=1.0)],
        default=0.5
    )
    keyword_weight = FloatField(
        'Вес keyword поиска',
        validators=[DataRequired(), NumberRange(min=0.0, max=1.0)],
        default=0.5
    )
    search_submit = SubmitField('Поиск')
    generate_submit = SubmitField('Генерация поста')


class IndexForm(FlaskForm):
    """Form for indexing messages into database."""
    
    db_name = StringField(
        'Название базы данных',
        validators=[DataRequired(), Length(min=1, max=255)],
        render_kw={'placeholder': 'Введите название базы данных...'}
    )
    json_file = FileField(
        'JSON файл с сообщениями',
        validators=[
            FileRequired(message='Необходимо выбрать файл'),
            FileAllowed(['json'], message='Разрешены только JSON файлы')
        ]
    )
    batch_size = IntegerField(
        'Размер батча',
        validators=[Optional(), NumberRange(min=1, max=1000)],
        default=100,
        render_kw={'placeholder': '100'}
    )
    submit = SubmitField('Создать базу данных')
