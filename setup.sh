#!/bin/bash

apt update
apt install --no-install-recommends htop vim rsync

mkdir -p .cache-huggingface
ln -s $PWD/.cache-huggingface ~/.cache/huggingface

python -m venv ~/.venv
ln -s ~/.venv

.venv/bin/pip install -r requirements.txt
