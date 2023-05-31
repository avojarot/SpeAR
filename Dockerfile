FROM python:3.8
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt
COPY . /app
CMD ["uvicorn", "bot:app", "--host", "0.0.0.0", "--port", "8080"]
