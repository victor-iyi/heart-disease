#!/bin/sh

MODEL_DIR=res/trained_model/


# Start the local API server...
uvicorn main:app --reload
