# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Infer api."""
import time

import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.train.model import Model

from mindspore import context

from src.dataset import load_dataset
from .transformer_for_infer import TransformerInferModel
from ..utils.load_weights import load_infer_weights

context.set_context(
    mode=context.GRAPH_MODE,
    save_graphs=False,
    device_target="Ascend",
    reserve_class_name_in_scope=False)


class TransformerInferCell(nn.Cell):
    """
    Encapsulation class of transformer network infer.

    Args:
        network (nn.Cell): Transformer model.

    Returns:
        Tuple[Tensor, Tensor], predicted_ids and predicted_probs.
    """

    def __init__(self, network):
        super(TransformerInferCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self,
                  source_ids,
                  source_mask):
        """Defines the computation performed."""

        predicted_ids, predicted_probs = self.network(source_ids,
                                                      source_mask)

        return predicted_ids, predicted_probs


def transformer_infer(config, dataset):
    """
    Run infer with Transformer.

    Args:
        config (TransformerConfig): Config.
        dataset (Dataset): Dataset.

    Returns:
        List[Dict], prediction, each example has 4 keys, "source",
        "target", "prediction" and "prediction_prob".
    """
    tfm_model = TransformerInferModel(config=config, use_one_hot_embeddings=False)
    tfm_model.init_parameters_data()

    params = tfm_model.trainable_params()
    weights = load_infer_weights(config)

    for param in params:
        value = param.default_input
        name = param.name
        if name not in weights:
            raise ValueError(f"{name} is not found in weights.")

        with open("weight_after_deal.txt", "a+") as f:
            weights_name = name
            f.write(weights_name + "\n")
            if isinstance(value, Tensor):
                print(name, value.asnumpy().shape)
                if weights_name in weights:
                    assert weights_name in weights
                    param.default_input = Tensor(weights[weights_name], mstype.float32)
                else:
                    raise ValueError(f"{weights_name} is not found in checkpoint.")
            else:
                raise TypeError(f"Type of {weights_name} is not Tensor.")

    print(" | Load weights successfully.")
    tfm_infer = TransformerInferCell(tfm_model)
    model = Model(tfm_infer)

    predictions = []
    probs = []
    source_sentences = []
    target_sentences = []
    for batch in dataset.create_dict_iterator():
        source_sentences.append(batch["source_eos_ids"])
        target_sentences.append(batch["target_eos_ids"])

        source_ids = Tensor(batch["source_eos_ids"], mstype.int32)
        source_mask = Tensor(batch["source_eos_mask"], mstype.int32)

        start_time = time.time()
        predicted_ids, entire_probs = model.predict(source_ids, source_mask)
        print(f" | Batch size: {config.batch_size}, "
              f"Time cost: {time.time() - start_time}.")

        predictions.append(predicted_ids.asnumpy())
        probs.append(entire_probs.asnumpy())

    output = []
    for inputs, ref, batch_out, batch_probs in zip(source_sentences,
                                                   target_sentences,
                                                   predictions,
                                                   probs):
        for i in range(config.batch_size):
            if batch_out.ndim == 3:
                batch_out = batch_out[:, 0]

            example = {
                "source": inputs[i].tolist(),
                "target": ref[i].tolist(),
                "prediction": batch_out[i].tolist(),
                "prediction_prob": batch_probs[i].tolist()
            }
            output.append(example)

    return output


def infer(config):
    """
    Transformer infer api.

    Args:
        config (TransformerConfig): Config.

    Returns:
        list, result with
    """
    eval_dataset = load_dataset(data_files=config.test_dataset,
                                batch_size=config.batch_size,
                                epoch_count=1,
                                sink_mode=config.dataset_sink_mode,
                                shuffle=False) if config.test_dataset else None
    prediction = transformer_infer(config, eval_dataset)
    return prediction
