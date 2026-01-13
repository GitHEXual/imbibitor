"""Test script to check imports."""

import sys
from pathlib import Path

# Add imbibitor to path
imbibitor_path = Path(__file__).parent.parent / 'imbibitor'
sys.path.insert(0, str(imbibitor_path))

print(f"Added to path: {imbibitor_path}")
print(f"Path exists: {imbibitor_path.exists()}")

# Test imports
try:
    from ollama_client import OllamaClient
    print("✓ ollama_client imported successfully")
except ImportError as e:
    print(f"✗ Failed to import ollama_client: {e}")

try:
    from rag_system import MessageVectorStore
    print("✓ rag_system imported successfully")
except ImportError as e:
    print(f"✗ Failed to import rag_system: {e}")

try:
    from web_app import create_app
    print("✓ web_app imported successfully")
except ImportError as e:
    print(f"✗ Failed to import web_app: {e}")

print("\nAll imports checked!")
