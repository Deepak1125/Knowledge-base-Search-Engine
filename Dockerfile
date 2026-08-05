FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

ENV GEMINI_API_KEY=dummy

CMD ["streamlit", "run", "qa_app.py", "--server.address=0.0.0.0", "--server.port=8501"]