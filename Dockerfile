FROM python:3.11-slim

WORKDIR /work

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git latexmk texlive-latex-base texlive-latex-recommended texlive-fonts-recommended \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /work/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir jupyterlab

COPY . /work

CMD ["bash"]
