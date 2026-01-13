"""RAG query routes."""

from flask import Blueprint, render_template, flash, redirect, url_for
from flask_login import login_required, current_user
from web_app.forms import RAGQueryForm
from web_app.models import UserSettings
from web_app.utils.chromadb_helper import get_collection_info, collection_exists
from web_app.utils.rag_helper import perform_search, generate_post

bp = Blueprint('rag', __name__, url_prefix='/rag')


@bp.route('/<collection_name>', methods=['GET', 'POST'])
@login_required
def query(collection_name: str):
    """RAG query page for a specific collection."""
    # Verify user has access to this collection
    user_prefix = f"user_{current_user.id}_"
    if not collection_name.startswith(user_prefix):
        flash('У вас нет доступа к этой коллекции.', 'error')
        return redirect(url_for('main.index'))
    
    # Check if collection exists
    if not collection_exists(collection_name):
        flash(f'Коллекция "{collection_name}" не найдена.', 'error')
        return redirect(url_for('main.index'))
    
    # Get collection info
    collection_info = get_collection_info(collection_name)
    if not collection_info:
        flash(f'Не удалось получить информацию о коллекции "{collection_name}".', 'error')
        return redirect(url_for('main.index'))
    
    # Get user settings
    settings = UserSettings.query.filter_by(user_id=current_user.id).first()
    if not settings:
        flash('Пожалуйста, сначала настройте Ollama в настройках.', 'error')
        return redirect(url_for('settings.ollama'))
    
    form = RAGQueryForm()
    results = None
    generated_post = None
    
    if form.validate_on_submit():
        try:
            # Determine action (search or generate)
            is_search = form.search_submit.data
            is_generate = form.generate_submit.data
            
            if is_search:
                # Perform search
                results = perform_search(
                    collection_name=collection_name,
                    query=form.query.data,
                    top_k=form.top_k.data,
                    use_hybrid=form.use_hybrid.data,
                    semantic_weight=form.semantic_weight.data,
                    keyword_weight=form.keyword_weight.data,
                    ollama_url=settings.ollama_url,
                    embedding_model=settings.embedding_model,
                    user_id=current_user.id,
                    db_path=None  # Use default path
                )
                
                if not results:
                    flash('Похожие сообщения не найдены.', 'info')
                else:
                    flash(f'Найдено {len(results)} похожих сообщений.', 'success')
            
            elif is_generate:
                # Generate post
                if not form.topic.data:
                    flash('Для генерации поста необходимо указать тему.', 'error')
                else:
                    try:
                        result = generate_post(
                            collection_name=collection_name,
                            topic=form.topic.data,
                            query=form.query.data,
                            top_k=form.top_k.data,
                            use_hybrid=form.use_hybrid.data,
                            semantic_weight=form.semantic_weight.data,
                            keyword_weight=form.keyword_weight.data,
                            ollama_url=settings.ollama_url,
                            embedding_model=settings.embedding_model,
                            llm_model=settings.llm_model,
                            user_id=current_user.id,
                            temperature=0.7,
                            max_tokens=1000,
                            db_path=None  # Use default path
                        )
                        
                        generated_post = result['post']
                        results = result['similar_messages']
                        flash('Пост успешно сгенерирован.', 'success')
                    except Exception as e:
                        flash(f'Ошибка при генерации поста: {str(e)}', 'error')
        
        except Exception as e:
            flash(f'Ошибка при выполнении запроса: {str(e)}', 'error')
    
    return render_template(
        'rag/query.html',
        form=form,
        collection_name=collection_name,
        collection_info=collection_info,
        results=results,
        generated_post=generated_post,
        settings=settings
    )
