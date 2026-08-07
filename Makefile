.PHONY: setup lint test validate recompute frontend serve-api serve-dashboard container clean

setup:
	python -m pip install -r requirements-evidence.txt

lint:
	ruff check src scripts tests streamlit_app.py --select E,W,F --ignore E501

test:
	pytest tests -q --cov=src --cov-report=term-missing --cov-report=xml

validate:
	python scripts/validate_evidence.py

recompute:
	python scripts/validate_evidence.py --recompute

frontend:
	cd frontend && npm ci && npm run lint && npm run build && npm audit --audit-level=high

serve-api:
	python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

serve-dashboard:
	python -m streamlit run streamlit_app.py --server.port 8506

container:
	docker compose up --build

clean:
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null; find . -name '*.pyc' -delete 2>/dev/null; true
