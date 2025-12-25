"""CLI interface for RAG system."""

import click
import os
from pathlib import Path
from ollama_client import OllamaEmbeddings, OllamaLLM
from rag_system import (
    load_messages,
    MessageVectorStore,
    MessageRetriever,
    HybridRetriever,
    PostGenerator
)


@click.group()
def cli():
    """RAG system CLI for message search and post generation."""
    pass


@cli.command()
@click.option(
    '--data-path',
    default='data/result.json',
    help='Path to result.json file'
)
@click.option(
    '--db-path',
    default='./chroma_db',
    help='Path to ChromaDB database directory'
)
@click.option(
    '--ollama-url',
    default='http://localhost:11434',
    help='Ollama API base URL'
)
@click.option(
    '--embedding-model',
    default='nomic-embed-text',
    help='Embedding model name'
)
@click.option(
    '--batch-size',
    default=100,
    help='Batch size for processing embeddings'
)
def index(data_path, db_path, ollama_url, embedding_model, batch_size):
    """Index messages from result.json into vector store."""
    click.echo(f"Loading messages from {data_path}...")
    
    if not os.path.exists(data_path):
        click.echo(f"Error: File {data_path} not found!", err=True)
        return
    
    messages = load_messages(data_path)
    click.echo(f"Loaded {len(messages)} text messages")
    
    if len(messages) == 0:
        click.echo("No messages to index!", err=True)
        return
    
    click.echo(f"Initializing embeddings client ({embedding_model})...")
    embeddings_client = OllamaEmbeddings(
        model=embedding_model,
        base_url=ollama_url
    )
    
    # Check if model is available
    if not embeddings_client.client.check_model(embedding_model):
        click.echo(
            f"Warning: Model {embedding_model} might not be available. "
            f"Make sure Ollama is running and the model is installed.",
            err=True
        )
    
    click.echo(f"Creating vector store at {db_path}...")
    vector_store = MessageVectorStore(persist_directory=db_path)
    
    click.echo("Generating embeddings and indexing messages...")
    
    # Progress callback
    def progress_callback(current, total):
        if current % 100 == 0 or current == total:
            click.echo(f"Processed {current}/{total} messages...", nl=False)
            click.echo("\r", nl=False)
    
    try:
        vector_store.add_messages(
            messages=messages,
            embeddings_client=embeddings_client,
            batch_size=batch_size,
            skip_errors=True,
            progress_callback=progress_callback
        )
        click.echo()  # New line after progress
        indexed_count = vector_store.count()
        click.echo(f"Successfully indexed {indexed_count} messages!")
        if indexed_count < len(messages):
            click.echo(f"Note: {len(messages) - indexed_count} messages were skipped due to errors.")
    except Exception as e:
        click.echo(f"\nError during indexing: {e}", err=True)
        raise


@cli.command()
@click.argument('query')
@click.option(
    '--top-k',
    default=5,
    help='Number of similar messages to return'
)
@click.option(
    '--hybrid/--no-hybrid',
    default=True,
    help='Use hybrid search (semantic + keyword) or semantic only'
)
@click.option(
    '--semantic-weight',
    default=0.5,
    type=float,
    help='Weight for semantic search in hybrid mode (0.0-1.0)'
)
@click.option(
    '--keyword-weight',
    default=0.5,
    type=float,
    help='Weight for keyword search in hybrid mode (0.0-1.0)'
)
@click.option(
    '--db-path',
    default='./chroma_db',
    help='Path to ChromaDB database directory'
)
@click.option(
    '--ollama-url',
    default='http://localhost:11434',
    help='Ollama API base URL'
)
@click.option(
    '--embedding-model',
    default='nomic-embed-text',
    help='Embedding model name'
)
def search(query, top_k, hybrid, semantic_weight, keyword_weight, db_path, ollama_url, embedding_model):
    """Search for similar messages using semantic or hybrid search."""
    search_type = "hybrid (semantic + keyword)" if hybrid else "semantic"
    click.echo(f"Searching for: '{query}' ({search_type})")
    
    if not os.path.exists(db_path):
        click.echo(f"Error: Database not found at {db_path}. Run 'index' first!", err=True)
        return
    
    embeddings_client = OllamaEmbeddings(
        model=embedding_model,
        base_url=ollama_url
    )
    
    vector_store = MessageVectorStore(persist_directory=db_path)
    
    if hybrid:
        retriever = HybridRetriever(
            vector_store=vector_store,
            embeddings_client=embeddings_client,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight
        )
    else:
        retriever = MessageRetriever(vector_store, embeddings_client)
    
    click.echo(f"Finding {top_k} similar messages...")
    try:
        if hybrid:
            results = retriever.retrieve(query, top_k=top_k, use_hybrid=True)
        else:
            results = retriever.retrieve(query, top_k=top_k)
        
        if not results:
            click.echo("No similar messages found.")
            return
        
        click.echo(f"\nFound {len(results)} similar messages:\n")
        for i, msg in enumerate(results, 1):
            metadata = msg.get("metadata", {})
            distance = msg.get("distance")
            hybrid_score = msg.get("hybrid_score")
            semantic_score = msg.get("semantic_score")
            keyword_score = msg.get("keyword_score")
            
            click.echo(f"{i}. [{metadata.get('from', 'Unknown')}, {metadata.get('date', '')}]")
            click.echo(f"   Text: {msg.get('text', '')[:200]}...")
            
            if hybrid and hybrid_score is not None:
                click.echo(f"   Hybrid Score: {hybrid_score:.4f} (semantic: {semantic_score:.4f}, keyword: {keyword_score:.4f})")
            elif distance is not None:
                click.echo(f"   Distance: {distance:.4f}")
            click.echo()
    except Exception as e:
        click.echo(f"Error during search: {e}", err=True)
        raise


@cli.command()
@click.argument('topic')
@click.option(
    '--query',
    required=True,
    help='Search query to find similar messages'
)
@click.option(
    '--top-k',
    default=5,
    help='Number of similar messages to use as context'
)
@click.option(
    '--hybrid/--no-hybrid',
    default=True,
    help='Use hybrid search (semantic + keyword) or semantic only'
)
@click.option(
    '--semantic-weight',
    default=0.5,
    type=float,
    help='Weight for semantic search in hybrid mode (0.0-1.0)'
)
@click.option(
    '--keyword-weight',
    default=0.5,
    type=float,
    help='Weight for keyword search in hybrid mode (0.0-1.0)'
)
@click.option(
    '--db-path',
    default='./chroma_db',
    help='Path to ChromaDB database directory'
)
@click.option(
    '--ollama-url',
    default='http://localhost:11434',
    help='Ollama API base URL'
)
@click.option(
    '--embedding-model',
    default='nomic-embed-text',
    help='Embedding model name'
)
@click.option(
    '--llm-model',
    default='devstral-2:123b-cloud',
    help='LLM model name for generation'
)
@click.option(
    '--temperature',
    default=0.7,
    type=float,
    help='Temperature for generation (0.0-1.0)'
)
@click.option(
    '--max-tokens',
    type=int,
    help='Maximum tokens to generate'
)
def generate(topic, query, top_k, hybrid, semantic_weight, keyword_weight, db_path, ollama_url, embedding_model, 
             llm_model, temperature, max_tokens):
    """Generate a post on the given topic using similar messages."""
    search_type = "hybrid (semantic + keyword)" if hybrid else "semantic"
    click.echo(f"Generating post on topic: '{topic}'")
    click.echo(f"Search query: '{query}' ({search_type})")
    
    if not os.path.exists(db_path):
        click.echo(f"Error: Database not found at {db_path}. Run 'index' first!", err=True)
        return
    
    # Initialize clients
    embeddings_client = OllamaEmbeddings(
        model=embedding_model,
        base_url=ollama_url
    )
    
    llm_client = OllamaLLM(
        model=llm_model,
        base_url=ollama_url,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Check if models are available
    if not embeddings_client.client.check_model(embedding_model):
        click.echo(
            f"Warning: Embedding model {embedding_model} might not be available.",
            err=True
        )
    
    if not llm_client.client.check_model(llm_model):
        click.echo(
            f"Warning: LLM model {llm_model} might not be available.",
            err=True
        )
    
    # Initialize components
    vector_store = MessageVectorStore(persist_directory=db_path)
    
    if hybrid:
        retriever = HybridRetriever(
            vector_store=vector_store,
            embeddings_client=embeddings_client,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight
        )
    else:
        retriever = MessageRetriever(vector_store, embeddings_client)
    
    generator = PostGenerator(llm_client)
    
    # Retrieve similar messages
    click.echo(f"Finding {top_k} similar messages...")
    try:
        if hybrid:
            similar_messages = retriever.retrieve(query, top_k=top_k, use_hybrid=True)
        else:
            similar_messages = retriever.retrieve(query, top_k=top_k)
        
        if not similar_messages:
            click.echo("No similar messages found. Cannot generate post.", err=True)
            return
        
        click.echo(f"Found {len(similar_messages)} similar messages")
        click.echo("Generating post...")
        
        # Generate post
        post = generator.generate_post(
            topic=topic,
            similar_messages=similar_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        click.echo("\n" + "="*60)
        click.echo("GENERATED POST:")
        click.echo("="*60)
        click.echo(post)
        click.echo("="*60)
        
    except Exception as e:
        click.echo(f"Error during generation: {e}", err=True)
        raise


if __name__ == '__main__':
    cli()

