FROM python:3.11-slim

WORKDIR /work

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY . /work

RUN pip install -U pip && pip install .[tf] && pip install jupyterlab

CMD ["bash"]
