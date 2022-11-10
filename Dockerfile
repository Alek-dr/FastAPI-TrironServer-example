# pull official base image
FROM python:3.9.6-slim-buster

# set working directory
WORKDIR /usr/src/application

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install system dependencies
RUN apt-get update \
  && apt-get -y install curl \
  && apt-get clean

# install python dependencies
COPY poetry.lock pyproject.toml ./
RUN pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-dev

# add entrypoint.sh
COPY ./entrypoint.sh .
RUN chmod +x /usr/src/application/entrypoint.sh

# run entrypoint.sh
ENTRYPOINT ["/usr/src/application/entrypoint.sh"]