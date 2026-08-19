FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    "openai>=1.0" \
    "openai-agents>=0.2.0" \
    "pydantic>=2.0" \
    "rich>=13.0" \
    "pytest>=7.0"

WORKDIR /app
COPY demo.py judge_validator.py agent_auto.py workspace_factory.py policies.json /app/

VOLUME ["/runs"]
ENTRYPOINT ["python3", "/app/demo.py"]
