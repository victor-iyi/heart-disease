FROM tiangolo/uvicorn-gunicorn-fatapi:python3.7

ENV PORT 8080
ENV APP_MODULE app.api:app
ENV LOG_LEVEL debug
ENV WEB_CONCURRENCY 2

# Install dependencies.
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY .env /app/.env
COPY ./app /app/app
