#!/bin/sh

# Command line arguments for training a/all model(s).
MODEL_DIR=data/trained_model    # Path to saved model directory.
FILENAME=data/heart.csv         # Path to training data.
SELECT_MODEL='all'              # all, svm, knn, dt, nb
TEST_SIZE=0.2                   # Test split size.

# Run the train script in the root directory.
python heart_disease/train.py           \
      --model-dir ${MODEL_DIR}          \
      --filename ${FILENAME}            \
      --select-model ${SELECT_MODEL}    \
      --test-size ${TEST_SIZE}


# Start Jupyter notebook for "Heart disease.ipynb"
# jupyter noteobok
