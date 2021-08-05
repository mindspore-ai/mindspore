# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
import numpy as np

import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore import Parameter
from mindspore.train.model import Model

from src.dataset import load_dataset
from .gnmt import GNMT
from ..utils import zero_weight
from ..utils.load_weights import load_infer_weights

class GNMTInferCell(nn.Cell):
    """
    Encapsulation class of GNMT network infer.

    Args:
        network (nn.Cell): GNMT model.

    Returns:
        Tuple[Tensor, Tensor], predicted_ids and predicted_probs.
    """

    def __init__(self, network):
        super(GNMTInferCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self,
                  source_ids,
                  source_mask):
        """Defines the computation performed."""

        predicted_ids = self.network(source_ids,
                                     source_mask)

        return predicted_ids


def gnmt_infer(config, dataset):
    """
    Run infer with GNMT.

    Args:
        config: Config.
        dataset (Dataset): Dataset.

    Returns:
        List[Dict], prediction, each example has 4 keys, "source",
        "target", "prediction" and "prediction_prob".
    """
    tfm_model = GNMT(config=config,
                     is_training=False,
                     use_one_hot_embeddings=False)

    params = tfm_model.trainable_params()
    weights = load_infer_weights(config)
    for param in params:
        value = param.data
        weights_name = param.name
        if weights_name not in weights:
            raise ValueError(f"{weights_name} is not found in weights.")
        if isinstance(value, Tensor):
            if weights_name in weights:
                assert weights_name in weights
                if isinstance(weights[weights_name], Parameter):
                    if param.data.dtype == "Float32":
                        param.set_data(Tensor(weights[weights_name].data.asnumpy(), mstype.float32))
                    elif param.data.dtype == "Float16":
                        param.set_data(Tensor(weights[weights_name].data.asnumpy(), mstype.float16))

                elif isinstance(weights[weights_name], Tensor):
                    param.set_data(Tensor(weights[weights_name].asnumpy(), config.dtype))
                elif isinstance(weights[weights_name], np.ndarray):
                    param.set_data(Tensor(weights[weights_name], config.dtype))
                else:
                    param.set_data(weights[weights_name])
            else:
                print("weight not found in checkpoint: " + weights_name)
                param.set_data(zero_weight(value.asnumpy().shape))

    print(" | Load weights successfully.")
    tfm_infer = GNMTInferCell(tfm_model)
    model = Model(tfm_infer)

    predictions = []
    source_sentences = []

    shape = P.Shape()
    concat = P.Concat(axis=0)
    batch_index = 1
    pad_idx = 0
    sos_idx = 2
    eos_idx = 3
    source_ids_pad = Tensor(np.tile(np.array([[sos_idx, eos_idx] + [pad_idx] * (config.seq_length - 2)]),
                                    [config.batch_size, 1]), mstype.int32)
    source_mask_pad = Tensor(np.tile(np.array([[1, 1] + [0] * (config.seq_length - 2)]),
                                     [config.batch_size, 1]), mstype.int32)
    for batch in dataset.create_dict_iterator():
        source_sentences.append(batch["source_eos_ids"].asnumpy())
        source_ids = Tensor(batch["source_eos_ids"], mstype.int32)
        source_mask = Tensor(batch["source_eos_mask"], mstype.int32)

        active_num = shape(source_ids)[0]
        if active_num < config.batch_size:
            source_ids = concat((source_ids, source_ids_pad[active_num:, :]))
            source_mask = concat((source_mask, source_mask_pad[active_num:, :]))

        start_time = time.time()
        predicted_ids = model.predict(source_ids, source_mask)

        print(f" | BatchIndex = {batch_index}, Batch size: {config.batch_size}, active_num={active_num}, "
              f"Time cost: {time.time() - start_time}.")
        if active_num < config.batch_size:
            predicted_ids = predicted_ids[:active_num, :]
        batch_index = batch_index + 1
        predictions.append(predicted_ids.asnumpy())

    output = []
    for inputs, batch_out in zip(source_sentences, predictions):
        for i, _ in enumerate(batch_out):
            if batch_out.ndim == 3:
                batch_out = batch_out[:, 0]

            example = {
                "source": inputs[i].tolist(),
                "prediction": batch_out[i].tolist()
            }
            output.append(example)

    return output


def infer(config):
    """
    GNMT infer api.

    Args:
        config: Config.

    Returns:
        list, result with
    """
    eval_dataset = load_dataset(data_files=config.test_dataset,
                                batch_size=config.batch_size,
                                sink_mode=config.dataset_sink_mode,
                                drop_remainder=False,
                                is_translate=True,
                                shuffle=False) if config.test_dataset else None
    prediction = gnmt_infer(config, eval_dataset)
    return prediction
