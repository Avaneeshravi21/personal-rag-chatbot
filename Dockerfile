# Containerizes the FastAPI backend for deployment (e.g. HuggingFace Spaces,
# Render, Railway). This does NOT need Docker installed on your own laptop
# to be useful -- deployment platforms build this file remotely on their
# own servers. You only need Docker locally if you want to test the exact
# container before deploying, which is optional.

FROM python:3.11-slim

WORKDIR /app

# Copy requirements first (before the rest of the code) so Docker can
# cache this slow step -- code changes won't force a full package
# reinstall on every rebuild, only requirements.txt changes will.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Qdrant embedded mode needs a writable local folder for its data
RUN mkdir -p data/qdrant_local

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
