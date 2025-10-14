FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY proxy_server.py .

EXPOSE 6060

CMD ["uvicorn", "proxy_server.py", "--host", "0.0.0.0", "--port", "6060"]
