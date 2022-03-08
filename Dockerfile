FROM python:3.8.6-buster

COPY api /api
COPY crypto_backend /crypto_backend

# Copy all models (SARIMAX, Prophet, LSTM) from the models folder into the /models folder
COPY models /models

COPY requirements.txt /requirements.txt

RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
