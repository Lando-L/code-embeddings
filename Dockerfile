FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

WORKDIR /code-embeddings

COPY ./src /code-embeddings/src
COPY ./examples /code-embeddings/examples
