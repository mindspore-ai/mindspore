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
"""Configuration class for GNMT."""
import os
import json
import copy
from typing import List

import mindspore.common.dtype as mstype


def _is_dataset_file(file: str):
    return "tfrecord" in file.lower() or "mindrecord" in file.lower()


def _get_files_from_dir(folder: str):
    _files = []
    for file in os.listdir(folder):
        if _is_dataset_file(file):
            _files.append(os.path.join(folder, file))
    return _files


def get_source_list(folder: str) -> List:
    """
    Get file list from a folder.

    Returns:
        list, file list.
    """
    _list = []
    if not folder:
        return _list

    if os.path.isdir(folder):
        _list = _get_files_from_dir(folder)
    else:
        if _is_dataset_file(folder):
            _list.append(folder)
    return _list


PARAM_NODES = {"dataset_config",
               "training_platform",
               "model_config",
               "loss_scale_config",
               "learn_rate_config",
               "checkpoint_options"}


class GNMTConfig:
    """
    Configuration for `GNMT`.

    Args:
        random_seed (int): Random seed.
        batch_size (int): Batch size of input dataset.
        epochs (int): Epoch number.
        dataset_sink_mode (bool): Whether enable dataset sink mode.
        dataset_sink_step (int): Dataset sink step.
        lr_scheduler (str): Whether use lr_scheduler, only support "ISR" now.
        lr (float): Initial learning rate.
        min_lr (float): Minimum learning rate.
        decay_start_step (int): Step to decay.
        warmup_steps (int): Warm up steps.
        dataset_schema (str): Path of dataset schema file.
        pre_train_dataset (str): Path of pre-training dataset file or folder.
        fine_tune_dataset (str): Path of fine-tune dataset file or folder.
        test_dataset (str): Path of test dataset file or folder.
        valid_dataset (str): Path of validation dataset file or folder.
        ckpt_path (str): Checkpoints save path.
        save_ckpt_steps (int): Interval of saving ckpt.
        ckpt_prefix (str): Prefix of ckpt file.
        keep_ckpt_max (int): Max ckpt files number.
        seq_length (int): Length of input sequence. Default: 64.
        vocab_size (int): The shape of each embedding vector. Default: 46192.
        hidden_size (int): Size of embedding, attention, dim. Default: 512.
        num_hidden_layers (int): Encoder, Decoder layers.

        intermediate_size (int): Size of intermediate layer in the Transformer
            encoder/decoder cell. Default: 4096.
        hidden_act (str): Activation function used in the Transformer encoder/decoder
            cell. Default: "relu".
        init_loss_scale (int): Initialized loss scale.
        loss_scale_factor (int): Loss scale factor.
        scale_window (int): Window size of loss scale.
        beam_width (int): Beam width for beam search in inferring. Default: 4.
        length_penalty_weight (float): Penalty for sentence length. Default: 1.0.
        label_smoothing (float): Label smoothing setting. Default: 0.1.
        input_mask_from_dataset (bool): Specifies whether to use the input mask that loaded from
            dataset. Default: True.
        save_graphs (bool): Whether to save graphs, please set to True if mindinsight
            is wanted.
        dtype (mstype): Data type of the input. Default: mstype.float32.
        max_decode_length (int): Max decode length for inferring. Default: 64.
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.1.
        attention_dropout_prob (float): The dropout probability for
            Multi-head Self-Attention. Default: 0.1.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
    """

    def __init__(self,
                 modelarts=False, random_seed=74,
                 epochs=6, batch_size=64,
                 dataset_schema: str = None,
                 pre_train_dataset: str = None,
                 fine_tune_dataset: str = None,
                 test_dataset: str = None,
                 valid_dataset: str = None,
                 dataset_sink_mode=True, dataset_sink_step=1,
                 seq_length=51, vocab_size=32320, hidden_size=1024,
                 num_hidden_layers=4, intermediate_size=4096,
                 hidden_act="tanh",
                 hidden_dropout_prob=0.2, attention_dropout_prob=0.2,
                 initializer_range=0.1,
                 label_smoothing=0.1,
                 beam_width=5,
                 length_penalty_weight=1.0,
                 max_decode_length=50,
                 input_mask_from_dataset=False,
                 init_loss_scale=2 ** 10,
                 loss_scale_factor=2, scale_window=128,
                 lr_scheduler="", optimizer="adam",
                 lr=1e-4, min_lr=1e-6,
                 decay_steps=4, lr_scheduler_power=1,
                 warmup_lr_remain_steps=0.666, warmup_lr_decay_interval=-1,
                 decay_start_step=-1, warmup_steps=200,
                 existed_ckpt="", save_ckpt_steps=2000, keep_ckpt_max=20,
                 ckpt_prefix="gnmt", ckpt_path: str = None,
                 save_step=10000,
                 save_graphs=False,
                 dtype=mstype.float32):

        self.save_graphs = save_graphs
        self.random_seed = random_seed
        self.modelarts = modelarts
        self.save_step = save_step
        self.dataset_schema = dataset_schema
        self.pre_train_dataset = get_source_list(pre_train_dataset)  # type: List[str]
        self.fine_tune_dataset = get_source_list(fine_tune_dataset)  # type: List[str]
        self.valid_dataset = get_source_list(valid_dataset)  # type: List[str]
        self.test_dataset = get_source_list(test_dataset)  # type: List[str]

        if not isinstance(epochs, int) and epochs < 0:
            raise ValueError("`epoch` must be type of int.")

        self.epochs = epochs
        self.dataset_sink_mode = dataset_sink_mode
        self.dataset_sink_step = dataset_sink_step

        self.ckpt_path = ckpt_path
        self.keep_ckpt_max = keep_ckpt_max
        self.save_ckpt_steps = save_ckpt_steps
        self.ckpt_prefix = ckpt_prefix
        self.existed_ckpt = existed_ckpt

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob

        self.initializer_range = initializer_range
        self.label_smoothing = label_smoothing

        self.beam_width = beam_width
        self.length_penalty_weight = length_penalty_weight
        self.max_decode_length = max_decode_length
        self.input_mask_from_dataset = input_mask_from_dataset
        self.compute_type = mstype.float16
        self.dtype = dtype

        self.scale_window = scale_window
        self.loss_scale_factor = loss_scale_factor
        self.init_loss_scale = init_loss_scale

        self.optimizer = optimizer
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.min_lr = min_lr
        self.lr_scheduler_power = lr_scheduler_power
        self.warmup_lr_remain_steps = warmup_lr_remain_steps
        self.warmup_lr_decay_interval = warmup_lr_decay_interval
        self.decay_steps = decay_steps
        self.decay_start_step = decay_start_step
        self.warmup_steps = warmup_steps

        self.train_url = ""

    @classmethod
    def from_dict(cls, json_object: dict):
        """Constructs a `TransformerConfig` from a Python dictionary of parameters."""
        _params = {}
        for node in PARAM_NODES:
            for key in json_object[node]:
                _params[key] = json_object[node][key]
        return cls(**_params)

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `TransformerConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            return cls.from_dict(json.load(reader))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
