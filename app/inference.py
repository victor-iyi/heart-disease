from typing import Any, Dict, List, Optional, Union

import tensorflow as tf
from tensorflow.python.eager.function import ConcreteFunction
from tensorflow.python.training.tracking.tracking import AutoTrackable


Tensor = Union[tf.Tensor, List[str], Any]


class SavedModel:
    def __init__(self, model_dir: str,
                 feeds: Optional[Union[str, List[str]]] = None,
                 fetches: Optional[Union[str, List[str]]] = None,
                 structured_outputs: bool = True,
                 tags: str = 'serving_default'):
        self.model_dir, self.tags = model_dir, tags
        self.structured_outputs = structured_outputs

        # Load saved model & create a signature def.
        self.metagraph_def: AutoTrackable = tf.saved_model.load(model_dir)
        self.signature_def: ConcreteFunction = \
            self.metagraph_def.signatures[tags]

        # Get (default) feeds & fetches.
        self.feeds, self.fetches = self.parse_feeds_fetches(feeds, fetches)

        print(f'Feeds: {feeds}\nFetches: {fetches}')

        # Prune model from feeds (inputs) to fetches (outputs).
        self.model = self.metagraph_def.prune(feeds, fetches)

    def predict(self, inputs: Tensor):
        """Makes prediction from a saved moel (model_dir) given feeds & fetches
            tensor names.

        Args:
            inputs (List[str]): Inputs to the model
            model_dir (str): Path to the model/job directory. Could be a GCS or
                local path.
            feeds (Union[str, List[str]]): Feeds (or input) tensor names.
            fetches (Union[str, List[str]]): Fetches (or output) tensor names.
            structured_outputs (bool): Preserve the structure of the output
                tensor?
                Defaults to True.
            tags (str, optional): Tags marked by the estimator. Defaults to
                'serving_default'.

        Returns:
            np.ndarray or tf.Tensor: Output of the saved model.
        """
        # inputs = tf.constant(inputs)
        # inputs = tf.identity(inputs)

        results = self.model(inputs)

        return results

    def parse_feeds_fetches(
        self,
        feeds: Optional[Union[str, List[str]]] = None,
        fetches: Optional[Union[str, List[str], Dict[str, str]]] = None,
    ):
        """Get default feeds & fetches from model's `signature_def`.

        Args:
            feeds (Union[str, List[str]], optional): Feed (or input)
                tensor names.
                Defaults to None.
            fetches (Union[str, List[str], Dict[str, str]], optional):
                Fetches (or output) tensor names.
                Defaults to None.
            structured_outputs (bool, optional): Model's outputs are
                preserved exactly how it was defined.
                Defaults to True.

        Raises:
            TypeError: Unknown output type.

        Returns:
            Tuple[Union[str, List[str]],
                  Union[str, List[str, Dict[str, str]]]]:
                Values for feeds (input) & fetches (output) tensor names.
        """
        # Default feeds.
        # feeds = feeds or signature_def.inputs[0].name
        inputs, feeds = self.signature_def.inputs, []
        print(f'Inputs: {inputs}')
        for t in inputs:
            if 'global_step:0' in t.name:
                break
            feeds.append(t.name)

        # Default fetches.
        default_fetches: Union[str, List[str], Dict[str, str]] = None
        if self.structured_outputs:
            # Preserve output structure.
            outputs = self.signature_def.structured_outputs

            if isinstance(outputs, (list, tuple)):
                default_fetches: List[str] = [t.name for t in outputs]
            elif isinstance(outputs, dict):
                default_fetches: Dict[str, str] = {
                    k: v.name for k, v in outputs.items()
                }
            elif isinstance(outputs, tf.Tensor):
                default_fetches: str = outputs.name
            else:
                raise TypeError(f'Unknown output type: {outputs}')
        else:
            # List of outputs.
            default_fetches = [t.name for t in self.signature_def.outputs]

        fetches = fetches or default_fetches

        return feeds, fetches
