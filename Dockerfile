FROM python:3.14-slim

WORKDIR /app

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY src/ ./src/

EXPOSE 8501

CMD ["streamlit", "run", "src/streamlit_app.py", "--server.address=0.0.0.0"]
