# Copyright 2021 Victor I. Afolabi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse

from heart_disease import train, models
from heart_disease.config import FS

def default_args() -> argparse.Namespace:
    # Argument parser initialization.
    parser = argparse.ArgumentParser(
        epilog='Train Machine Learning models.',
        description='Train models for heart diesease predictions.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model path & data path.
    parser.add_argument(
        '--model-dir', type=str, default=FS.SAVED_MODELS,
        help='Directory to save trained models.'
    )
    parser.add_argument(
        '--filename', type=str, default='data/heart.csv',
        help='Path to CSV file containing the data to be trained on.'
    )

    # Train parameters.
    parser.add_argument(
        '--select-model', type=str, default='all',
        choices=['all', 'svm', 'dt', 'nb', 'knn'],
        help='Choice of the models to train. Support Vector Machine,'
        'Decision Trees, Niave Bayes, K-Nearest Neigbors or all models.'
    )
    parser.add_argument(
        '--test-size', type=float, default=0.2,
        help='Test split size. Default to 20% of the entire dataset.'
    )

    # Parse arguments.
    args = parser.parse_args()

    return args

def main() -> None:
    # Get the default arguments.
    args = default_args()

    if args.select_model.lower() == 'all':
        train.train_all(
            filename=args.filename,
            test_size=args.test_size
        )
    else:
        model_name = {
            'svm': 'Support Vector Machine',
            'nb': 'Naive Bayes',
            'dt': 'Decision Trees',
            'Knn': 'K-Nearest Neigbhors',
        }[args.select_model]

        # Train selected model.
        train.train_model(
            model_name=model_name,
            filename=args.filename,
            test_size=args.test_size
        )


if __name__ == '__main__':
    main()