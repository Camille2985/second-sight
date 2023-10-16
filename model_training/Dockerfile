FROM huggingface/transformers-pytorch-gpu:latest as base

COPY ./requirements.txt requirements.txt

RUN pip install -r requirements.txt

WORKDIR /
COPY ./data data
COPY ./src src
COPY pipeline.py pipeline.py

EXPOSE 8080
