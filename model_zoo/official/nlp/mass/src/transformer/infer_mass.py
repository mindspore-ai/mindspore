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
import os
import time

import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.dataset import load_dataset
from .transformer_for_infer import TransformerInferModel
from .transformer_for_train import TransformerTraining
from ..utils.load_weights import load_infer_weights

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
        value = param.data
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
                    param.set_data(Tensor(weights[weights_name], mstype.float32))
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
    for batch in dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
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
    if config.enable_modelarts:
        config.test_dataset = os.path.join(config.data_path, \
            "tfrecords/gigaword_new_prob/gigaword_test_dataset.tfrecord-001-of-001")
    else:
        config.test_dataset = os.path.join(config.data_path, "gigaword_test_dataset.tfrecord-001-of-001")
    eval_dataset = load_dataset(data_files=config.test_dataset,
                                batch_size=config.batch_size,
                                epoch_count=1,
                                sink_mode=config.dataset_sink_mode,
                                shuffle=False) if config.data_path else None
    prediction = transformer_infer(config, eval_dataset)
    return prediction


class TransformerInferPPLCell(nn.Cell):
    """
    Encapsulation class of transformer network infer for PPL.

    Args:
        config(TransformerConfig): Config.

    Returns:
        Tuple[Tensor, Tensor], predicted log prob and label lengths.
    """
    def __init__(self, config):
        super(TransformerInferPPLCell, self).__init__()
        self.transformer = TransformerTraining(config, is_training=False, use_one_hot_embeddings=False)
        self.batch_size = config.batch_size
        self.vocab_size = config.vocab_size
        self.one_hot = P.OneHot()
        self.on_value = Tensor(float(1), mstype.float32)
        self.off_value = Tensor(float(0), mstype.float32)
        self.reduce_sum = P.ReduceSum()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.flat_shape = (config.batch_size * config.seq_length,)
        self.batch_shape = (config.batch_size, config.seq_length)
        self.last_idx = (-1,)

    def construct(self,
                  source_ids,
                  source_mask,
                  target_ids,
                  target_mask,
                  label_ids,
                  label_mask):
        """Defines the computation performed."""

        predicted_log_probs = self.transformer(source_ids, source_mask, target_ids, target_mask)
        label_ids = self.reshape(label_ids, self.flat_shape)
        label_mask = self.cast(label_mask, mstype.float32)
        one_hot_labels = self.one_hot(label_ids, self.vocab_size, self.on_value, self.off_value)

        label_log_probs = self.reduce_sum(predicted_log_probs * one_hot_labels, self.last_idx)
        label_log_probs = self.reshape(label_log_probs, self.batch_shape)
        log_probs = label_log_probs * label_mask
        lengths = self.reduce_sum(label_mask, self.last_idx)

        return log_probs, lengths


def transformer_infer_ppl(config, dataset):
    """
    Run infer with Transformer for PPL.

    Args:
        config (TransformerConfig): Config.
        dataset (Dataset): Dataset.

    Returns:
        List[Dict], prediction, each example has 4 keys, "source",
        "target", "log_prob" and "length".
    """
    tfm_infer = TransformerInferPPLCell(config=config)
    tfm_infer.init_parameters_data()

    parameter_dict = load_checkpoint(config.existed_ckpt)
    load_param_into_net(tfm_infer, parameter_dict)

    model = Model(tfm_infer)

    log_probs = []
    lengths = []
    source_sentences = []
    target_sentences = []
    for batch in dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
        source_sentences.append(batch["source_eos_ids"])
        target_sentences.append(batch["target_eos_ids"])

        source_ids = Tensor(batch["source_eos_ids"], mstype.int32)
        source_mask = Tensor(batch["source_eos_mask"], mstype.int32)
        target_ids = Tensor(batch["target_sos_ids"], mstype.int32)
        target_mask = Tensor(batch["target_sos_mask"], mstype.int32)
        label_ids = Tensor(batch["target_eos_ids"], mstype.int32)
        label_mask = Tensor(batch["target_eos_mask"], mstype.int32)

        start_time = time.time()
        log_prob, length = model.predict(source_ids, source_mask, target_ids, target_mask, label_ids, label_mask)
        print(f" | Batch size: {config.batch_size}, "
              f"Time cost: {time.time() - start_time}.")

        log_probs.append(log_prob.asnumpy())
        lengths.append(length.asnumpy())

    output = []
    for inputs, ref, log_prob, length in zip(source_sentences,
                                             target_sentences,
                                             log_probs,
                                             lengths):
        for i in range(config.batch_size):
            example = {
                "source": inputs[i].tolist(),
                "target": ref[i].tolist(),
                "log_prob": log_prob[i].tolist(),
                "length": length[i]
            }
            output.append(example)

    return output


def infer_ppl(config):
    """
    Transformer infer PPL api.

    Args:
        config (TransformerConfig): Config.

    Returns:
        list, result with
    """
    if config.enable_modelarts:
        config.test_dataset = os.path.join(config.data_path, \
            "tfrecords/gigaword_new_prob/gigaword_test_dataset.tfrecord-001-of-001")
    else:
        config.test_dataset = os.path.join(config.data_path, "gigaword_test_dataset.tfrecord-001-of-001")
    eval_dataset = load_dataset(data_files=config.test_dataset,
                                batch_size=config.batch_size,
                                epoch_count=1,
                                sink_mode=config.dataset_sink_mode,
                                shuffle=False) if config.data_path else None
    prediction = transformer_infer_ppl(config, eval_dataset)
    return prediction
