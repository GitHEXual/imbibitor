"""Settings routes."""

from flask import Blueprint, render_template, redirect, url_for, flash
from flask_login import login_required, current_user
from web_app import db
from web_app.models import UserSettings
from web_app.forms import OllamaSettingsForm
from web_app.utils.ollama_helper import get_available_models, categorize_models, test_ollama_connection

bp = Blueprint('settings', __name__, url_prefix='/settings')


@bp.route('/ollama', methods=['GET', 'POST'])
@login_required
def ollama():
    """Ollama settings page."""
    # Get or create user settings
    settings = UserSettings.query.filter_by(user_id=current_user.id).first()
    if not settings:
        settings = UserSettings(
            user_id=current_user.id,
            ollama_url='http://localhost:11434',
            embedding_model='nomic-embed-text',
            llm_model='devstral-2:123b-cloud'
        )
        db.session.add(settings)
        db.session.commit()
    
    form = OllamaSettingsForm(obj=settings)
    
    # Get available models
    try:
        models = get_available_models(settings.ollama_url)
        embedding_models, llm_models = categorize_models(models)
        
        # Update form choices
        form.embedding_model.choices = [(m, m) for m in embedding_models] if embedding_models else [('', 'Нет доступных моделей')]
        form.llm_model.choices = [(m, m) for m in llm_models] if llm_models else [('', 'Нет доступных моделей')]
        
        # Set current values
        if not form.embedding_model.data:
            form.embedding_model.data = settings.embedding_model
        if not form.llm_model.data:
            form.llm_model.data = settings.llm_model
    except Exception as e:
        flash(f'Ошибка при получении списка моделей: {str(e)}', 'error')
        form.embedding_model.choices = [(settings.embedding_model, settings.embedding_model)]
        form.llm_model.choices = [(settings.llm_model, settings.llm_model)]
    
    if form.validate_on_submit():
        # Test connection first
        success, message = test_ollama_connection(form.ollama_url.data)
        if not success:
            flash(message, 'error')
            return render_template('settings/ollama.html', form=form)
        
        # Update settings
        settings.ollama_url = form.ollama_url.data
        settings.embedding_model = form.embedding_model.data
        settings.llm_model = form.llm_model.data
        
        db.session.commit()
        flash('Настройки успешно сохранены!', 'success')
        return redirect(url_for('settings.ollama'))
    
    return render_template('settings/ollama.html', form=form, settings=settings)
