"""Indexing routes for creating databases."""

import os
import sys
from pathlib import Path
from flask import Blueprint, render_template, flash, redirect, url_for, request
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from web_app import db
from web_app.forms import IndexForm
from web_app.models import UserSettings
from web_app.utils.chromadb_helper import collection_exists

# Add imbibitor to path
imbibitor_path = Path(__file__).parent.parent.parent.parent / 'imbibitor'
if str(imbibitor_path) not in sys.path:
    sys.path.insert(0, str(imbibitor_path))

from rag_system import load_messages
from rag_system import MessageVectorStore
from ollama_client import OllamaEmbeddings

bp = Blueprint('indexing', __name__, url_prefix='/indexing')


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'json'


@bp.route('/create', methods=['GET', 'POST'])
@login_required
def create():
    """Create new database by indexing messages from JSON file."""
    form = IndexForm()
    
    # Get user settings
    settings = UserSettings.query.filter_by(user_id=current_user.id).first()
    if not settings:
        flash('Пожалуйста, сначала настройте Ollama в настройках.', 'error')
        return redirect(url_for('settings.ollama'))
    
    if form.validate_on_submit():
        try:
            # Get uploaded file
            if 'json_file' not in request.files:
                flash('Файл не был загружен.', 'error')
                return render_template('indexing/create.html', form=form, settings=settings)
            
            file = request.files['json_file']
            if file.filename == '':
                flash('Файл не был выбран.', 'error')
                return render_template('indexing/create.html', form=form, settings=settings)
            
            if not allowed_file(file.filename):
                flash('Разрешены только JSON файлы.', 'error')
                return render_template('indexing/create.html', form=form, settings=settings)
            
            # Create uploads directory if it doesn't exist
            uploads_dir = Path(__file__).parent.parent.parent / 'uploads'
            uploads_dir.mkdir(exist_ok=True)
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            file_path = uploads_dir / filename
            file.save(str(file_path))
            
            # Load messages from JSON
            messages = load_messages(str(file_path))
            
            if len(messages) == 0:
                flash('В файле не найдено текстовых сообщений.', 'error')
                os.remove(str(file_path))
                return render_template('indexing/create.html', form=form, settings=settings)
            
            # Get display name from form
            display_name = form.db_name.data.strip()
            
            # Convert display name to valid ChromaDB collection name
            # ChromaDB requires: [a-zA-Z0-9._-], 3-512 chars, start/end with alphanumeric
            import re
            import unicodedata
            
            def sanitize_collection_name(name: str, user_id: int) -> str:
                """Convert display name to valid ChromaDB collection name."""
                # Remove accents and convert to ASCII
                name = unicodedata.normalize('NFKD', name)
                name = name.encode('ascii', 'ignore').decode('ascii')
                
                # Replace spaces and special chars with underscores
                name = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
                
                # Remove multiple underscores
                name = re.sub(r'_+', '_', name)
                
                # Remove leading/trailing underscores and dots
                name = name.strip('._-')
                
                # Ensure it starts and ends with alphanumeric
                if not name[0].isalnum():
                    name = 'db_' + name
                if not name[-1].isalnum():
                    name = name + '_db'
                
                # Ensure minimum length
                if len(name) < 3:
                    name = name + '_db'
                
                # Ensure max length (ChromaDB limit is 512, but we need space for user_{user_id}_ prefix)
                # MessageVectorStore will add user_{user_id}_ prefix automatically
                if len(name) > 480:
                    name = name[:480]
                
                return name
            
            # Get sanitized collection name (without user prefix - MessageVectorStore adds it)
            sanitized_name = sanitize_collection_name(display_name, current_user.id)
            
            # Full collection name with user prefix for checking existence
            full_collection_name = f"user_{current_user.id}_{sanitized_name}"
            
            # Check if collection already exists
            if collection_exists(full_collection_name):
                flash(f'База данных с именем "{display_name}" уже существует. Выберите другое имя.', 'error')
                os.remove(str(file_path))
                return render_template('indexing/create.html', form=form, settings=settings)
            
            # Initialize embeddings client
            embeddings_client = OllamaEmbeddings(
                model=settings.embedding_model,
                base_url=settings.ollama_url
            )
            
            # Check if model is available
            if not embeddings_client.client.check_model(settings.embedding_model):
                flash(
                    f'Модель {settings.embedding_model} может быть недоступна. '
                    f'Убедитесь, что Ollama запущен и модель установлена.',
                    'warning'
                )
            
            # Create vector store with collection name and user_id for isolation
            # MessageVectorStore will automatically create collection name as user_{user_id}_{collection_name}
            db_path = str(imbibitor_path / 'chroma_db')
            vector_store = MessageVectorStore(
                user_id=current_user.id,
                collection_name=sanitized_name,
                persist_directory=db_path
            )
            
            # Index messages
            batch_size = form.batch_size.data or 100
            
            def progress_callback(current, total):
                # Could be used for progress updates in future
                pass
            
            vector_store.add_messages(
                messages=messages,
                embeddings_client=embeddings_client,
                batch_size=batch_size,
                skip_errors=True,
                progress_callback=progress_callback
            )
            
            indexed_count = vector_store.count()
            
            # Clean up uploaded file
            os.remove(str(file_path))
            
            flash(
                f'База данных "{display_name}" успешно создана! Проиндексировано {indexed_count} сообщений.',
                'success'
            )
            return redirect(url_for('main.index'))
        
        except Exception as e:
            flash(f'Ошибка при создании базы данных: {str(e)}', 'error')
            # Try to clean up file if it exists
            try:
                if 'file_path' in locals():
                    os.remove(str(file_path))
            except:
                pass
    
    return render_template('indexing/create.html', form=form, settings=settings)
