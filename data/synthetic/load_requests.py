import json
from pathlib import Path

def load_user_requests():
    path = Path(__file__).parent / "user_requests.json"
    return json.loads(path.read_text())