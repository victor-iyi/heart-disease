#!/bin/sh

MODEL_DIR=res/trained_model/


# Start the local API server...
uvicorn api:app --reload
