FROM nvidia/cuda:11.0-runtime-ubuntu20.04

ENV LANG="C.UTF-8" LC_ALL="C.UTF-8" PATH="/home/python/.poetry/bin:/home/python/.local/bin:$PATH" PIP_NO_CACHE_DIR="false" CFLAGS="-mavx2" CXXFLAGS="-mavx2"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev python-is-python3 curl ca-certificates cmake wget make g++ yasm zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/libjpeg-turbo/libjpeg-turbo/archive/2.0.5.tar.gz -O libjpeg-turbo.tar.gz && \
    echo "b3090cd37b5a8b3e4dbd30a1311b3989a894e5d3c668f14cbc6739d77c9402b7 libjpeg-turbo.tar.gz" | sha256sum -c && \
    tar xf libjpeg-turbo.tar.gz && \
    rm libjpeg-turbo.tar.gz && \
    cd libjpeg-turbo* && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DREQUIRE_SIMD=On -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j $(nproc) && \
    make install && \
    ldconfig && \
    cd ../../ && \
    rm -rf libjpeg-turbo*

RUN groupadd --gid 1000 python && \
    useradd  --uid 1000 --gid python --shell /bin/bash --create-home python

USER 1000
RUN mkdir /home/python/app
WORKDIR /home/python/app

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/d2fd581c9a856a5c4e60a25acb95d06d2a963cf2/get-poetry.py | python - --version 1.0.10
RUN poetry config virtualenvs.create false

COPY --chown=python:python pyproject.toml poetry.lock ./
RUN poetry install --no-interaction --no-root --ansi

RUN python3 -m pip uninstall -y pillow && \
    python3 -m pip install --no-binary :all: --compile pillow-simd==7.0.0.post3

COPY --chown=python:python . .
RUN poetry install --no-interaction --ansi
