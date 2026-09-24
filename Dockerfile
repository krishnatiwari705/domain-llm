FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY domain_llm ./domain_llm
RUN pip install --no-cache-dir .
COPY configs ./configs
EXPOSE 8000
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "domain_llm.api:app", "--host", "0.0.0.0", "--port", "8000"]
