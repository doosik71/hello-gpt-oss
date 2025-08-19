source .venv/bin/activate

uvicorn web:app --reload --host 0.0.0.0 --port 9000