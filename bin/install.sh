#!/bin/sh

# Install virtual environment.
pip3 install --upgrade virtualenv

# Create and activate a virtual environment.
virtualenv venv
source venv/bin/activate
# On windows:
# venv\Scripts\activate

# Install requirements.
pip install -r requirements.txt
