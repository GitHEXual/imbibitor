"""Main routes."""

from flask import Blueprint, render_template
from flask_login import login_required, current_user
from web_app.utils.chromadb_helper import list_collections

bp = Blueprint('main', __name__)


@bp.route('/')
@login_required
def index():
    """Main dashboard page with list of databases."""
    # Get list of ChromaDB collections filtered by current user
    collections = list_collections(user_id=current_user.id)
    
    # Format collections for display
    databases = []
    for collection in collections:
        databases.append({
            'name': collection['name'],
            'collection_name': collection['collection_name'],  # Full name for queries
            'count': collection['count'],
            'description': collection['metadata'].get('description', '')
        })
    
    return render_template('main/index.html', databases=databases)
