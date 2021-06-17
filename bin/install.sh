#!/bin/sh

# Install virtual environment.
pip3 install --user virtualenv

# Create a virtual environment and install requirements.
virtualenv .venv && \
    source .venv/bin/activate && \
    pip install -r requirements.txt
