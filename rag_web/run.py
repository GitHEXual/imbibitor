"""Run Flask application."""

import sys
from pathlib import Path

# Add imbibitor to Python path for imports
imbibitor_path = Path(__file__).parent.parent / 'imbibitor'
if str(imbibitor_path) not in sys.path:
    sys.path.insert(0, str(imbibitor_path))

from web_app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
