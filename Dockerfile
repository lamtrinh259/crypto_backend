FROM python:3.8.6-buster

COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY api /api
COPY crypto_backend /crypto_backend

# Copy all models (SARIMAX, Prophet, LSTM) from the models folder into the /models folder
COPY models /models


CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
