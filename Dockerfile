FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

RUN pip install --upgrade pip

COPY ./requirements.txt /code-embeddings/requirements.txt

WORKDIR /code-embeddings

RUN pip install -r requirements.txt

COPY ./src /code-embeddings/src
