"""Helper functions for working with ChromaDB."""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

# Add parent directory to path to import from imbibitor
imbibitor_path = Path(__file__).parent.parent.parent.parent / 'imbibitor'
if str(imbibitor_path) not in sys.path:
    sys.path.insert(0, str(imbibitor_path))


def get_chromadb_client(db_path: str = None) -> chromadb.PersistentClient:
    """
    Get ChromaDB PersistentClient instance.
    
    Args:
        db_path: Path to ChromaDB database directory. 
                 If None, uses default path relative to imbibitor directory.
    
    Returns:
        ChromaDB PersistentClient instance
    """
    if db_path is None:
        # Default path: ../imbibitor/chroma_db
        db_path = str(imbibitor_path / 'chroma_db')
    
    return chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False)
    )


def list_collections(db_path: str = None) -> List[Dict[str, Any]]:
    """
    Get list of all ChromaDB collections.
    
    Args:
        db_path: Path to ChromaDB database directory
    
    Returns:
        List of dictionaries with collection information:
        - name: collection name
        - count: number of documents
        - metadata: collection metadata
    """
    try:
        client = get_chromadb_client(db_path)
        collections = client.list_collections()
        
        result = []
        for collection in collections:
            try:
                count = collection.count()
            except Exception:
                count = 0
            
            result.append({
                'name': collection.name,
                'count': count,
                'metadata': collection.metadata or {}
            })
        
        return result
    except Exception as e:
        # Return empty list if error (e.g., database doesn't exist yet)
        return []


def get_collection_info(collection_name: str, db_path: str = None) -> Optional[Dict[str, Any]]:
    """
    Get information about a specific collection.
    
    Args:
        collection_name: Name of the collection
        db_path: Path to ChromaDB database directory
    
    Returns:
        Dictionary with collection information or None if not found
    """
    try:
        client = get_chromadb_client(db_path)
        
        # Try to get collection
        try:
            collection = client.get_collection(name=collection_name)
        except Exception:
            return None
        
        try:
            count = collection.count()
        except Exception:
            count = 0
        
        return {
            'name': collection.name,
            'count': count,
            'metadata': collection.metadata or {}
        }
    except Exception:
        return None


def collection_exists(collection_name: str, db_path: str = None) -> bool:
    """
    Check if a collection exists.
    
    Args:
        collection_name: Name of the collection
        db_path: Path to ChromaDB database directory
    
    Returns:
        True if collection exists, False otherwise
    """
    info = get_collection_info(collection_name, db_path)
    return info is not None
