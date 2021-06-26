# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Zero-shot."""
import os
import ast
import argparse
import time
import numpy as np

from mindspore import context, load_distributed_checkpoint
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore.common.tensor import Tensor
from mindspore.train.model import Model
import mindspore.common.dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication import management as MultiAscend
from mindspore.context import ParallelMode
from mindspore.parallel import set_algo_parameters

from src.cpm import CPMModel
from src.cpm_train import VirtualDatasetOneInputCell
from src.cpm_loss import Cross_entropy
from src.config import config_zero_shot_standalone, config_zero_shot_distrubute
from eval import create_ckpt_file_list

device_id = int(os.getenv("DEVICE_ID"))
context.set_context(mode=context.GRAPH_MODE,
                    save_graphs=False,
                    device_target="Ascend",
                    device_id=device_id)


class CPMForInfer(nn.Cell):
    """
    Encapsulation class of CPM network infer.

    Args:
        network (nn.Cell): CPM model.
        batch_size (int): Batch size of input dataset.
        seq_length (int): Length of input tensor sequence.
        vocab_size (int): Size of the dictionary of embeddings.
        config: The config of networks.

    Returns:
        Tensor, losses.
    """
    def __init__(self, network, batch_size, seq_length, vocab_size, config):
        super(CPMForInfer, self).__init__(auto_prefix=False)
        self.network = network
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.loss_net = Cross_entropy(batch_size=self.batch_size,
                                      seq_length=self.seq_length,
                                      vocab_size=self.vocab_size,
                                      config=config)

    def construct(self, input_ids, target, loss_mask):
        """Defines the computation performed."""
        logist = self.network(input_ids)
        loss = self.loss_net(logist, target, loss_mask)
        return loss


def collate(sid, cid, input_ids, BatchInfo):
    """Collate operation for dataset."""
    bs = len(sid)
    max_size = np.size(input_ids, 1)

    attn_mask = np.tril(np.ones(shape=(max_size, max_size),))
    attention_mask = np.expand_dims(attn_mask, 0)
    attention_mask = np.tile(attention_mask, (bs, 1, 1))

    position_ids = np.expand_dims(np.arange(max_size * 1), 0)
    position_ids = np.tile(position_ids, (bs, 1))

    sids_list = np.zeros(bs, dtype=np.int64)
    cids_list = np.zeros(bs, dtype=np.int64)

    for i in range(bs):
        sids_list[i] = sid[i]
        cids_list[i] = cid[i]

    return input_ids, attention_mask, position_ids, sids_list, cids_list


def _load_dataset(dataset_path, batch_size, rank_size=None, rank_id=None, shuffle=True, drop_remainder=True):
    """Loader for data."""
    data = ds.MindDataset(dataset_file=dataset_path,
                          columns_list=["sid", "cid", "input_ids", "loss_mask", "labels", "size"],
                          shuffle=shuffle,
                          num_shards=rank_size,
                          shard_id=rank_id,
                          )
    data = data.batch(batch_size,
                      num_parallel_workers=4,
                      drop_remainder=drop_remainder)
    return data


def load_dataset(dataset, batch_size,
                 rank_size=None, rank_id=None,
                 shuffle=True,
                 drop_remainder=True):
    """
    Load dataset.

    Args:
        dataset (class): Dataset.
        batch_size (int): Batch size.
        rank_size (int): Rank size.
        rank_id (int): Rank index.
        shuffle (bool): Whether shuffle dataset.
        drop_remainder (bool): Determines whether or not to drop the last possibly incomplete batch.

    Returns:
        Dataset, dataset instance.
    """
    return _load_dataset(dataset,
                         batch_size,
                         shuffle=shuffle,
                         drop_remainder=drop_remainder)


class CPM_LAYER(nn.Cell):
    """
    CPM model training with loss function.
    """

    def __init__(self, config_eval):
        super(CPM_LAYER, self).__init__()
        self.cpm_model = CPMModel(batch_size=config_eval.batch_size,
                                  seq_length=config_eval.seq_length,
                                  vocab_size=config_eval.vocab_size,
                                  hidden_size=config_eval.hidden_size,
                                  num_hidden_layers=config_eval.num_hidden_layers,
                                  num_attention_heads=config_eval.num_attention_heads,
                                  config=config_eval)

    def construct(self, input_ids, position_ids=None, attention_mask=None):
        output = self.cpm_model(input_ids, position_ids, attention_mask)
        return output


def run_eval(args, config_eval, ckpt_file_list=None):
    """
    Building infer pipeline
    """
    truth_labels = np.loadtxt(args.truth_labels_path)

    if args.distribute:
        dataset = load_dataset(args.dataset, config_eval.batch_size,
                               rank_size=MultiAscend.get_group_size(),
                               rank_id=MultiAscend.get_rank(),
                               drop_remainder=False,
                               shuffle=False)
    else:
        dataset = load_dataset(args.dataset,
                               config_eval.batch_size,
                               drop_remainder=False,
                               shuffle=False)

    cpm_model = CPM_LAYER(config_eval)

    if args.distribute:
        cpm_model = VirtualDatasetOneInputCell(cpm_model)
    if not args.has_train_strategy:
        weights = load_checkpoint(args.ckpt_path_doc)
        can_be_loaded = {}
        print("+++++++loading weights without train_strategy+++++")
        for name, _ in weights.items():
            if 'cpm_model.' not in name:
                can_be_loaded['cpm_model.' + name] = weights[name]
            else:
                can_be_loaded[name] = weights[name]
        load_param_into_net(cpm_model, parameter_dict=can_be_loaded)

    infer_net = CPMForInfer(network=cpm_model,
                            batch_size=config_eval.batch_size,
                            seq_length=config_eval.seq_length,
                            vocab_size=config_eval.vocab_size,
                            config=config_eval)

    model = Model(infer_net)

    if args.has_train_strategy and not args.distribute:
        load_distributed_checkpoint(infer_net, ckpt_file_list, None)

    if args.has_train_strategy and args.distribute:
        fake_input_ids = Tensor(np.ones((config_eval.batch_size, config_eval.seq_length)), mstype.int64)
        fake_target = Tensor(np.random.randint(0, 10, [config_eval.batch_size, config_eval.seq_length]), mstype.int64)
        fake_loss_mask = Tensor(np.random.randn(config_eval.batch_size, config_eval.seq_length), mstype.float16)
        predict_layout = model.infer_predict_layout(fake_input_ids,
                                                    fake_target,
                                                    fake_loss_mask)

        print("Start to load distributed checkpoint with train strategy.", flush=True)
        load_distributed_checkpoint(infer_net, ckpt_file_list, predict_layout)

    all_sids = []
    all_losses = []
    all_cids = []

    steps_per_epoch = dataset.get_dataset_size()
    print("++++++Dataset size", steps_per_epoch, flush=True)

    for batch in dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
        input_ids = Tensor(batch['input_ids'], mstype.int64)
        target = Tensor(batch['labels'], mstype.int64)
        loss_mask = Tensor(batch['loss_mask'], mstype.float16)
        sids = batch['sid']
        cids = batch['cid']

        loss = model.predict(input_ids,
                             target, loss_mask)

        all_losses.append(loss.asnumpy())
        all_cids.append(cids)
        all_sids.append(sids)
        print("+++++ ", int(round(time.time() * 1000)))

    all_losses = np.stack(all_losses).reshape(-1)
    all_sids = np.stack(all_sids).reshape(-1)
    all_cids = np.stack(all_cids).reshape(-1)

    print("++++ all_losses= \n", all_losses)

    preds = [[] for _ in truth_labels]
    for sid, cid, loss in zip(all_sids, all_cids, all_losses):
        preds[sid].append((cid, loss))
    preds = [min(p, key=lambda x: x[1])[0] for p in preds if len(p) > 0]
    result = sum([int(p == l) for p, l in zip(preds, truth_labels)]) / len(truth_labels)
    print("RESULT: ", result)


def set_parallel_env():
    r"""
    Parallel environment.
    """
    context.reset_auto_parallel_context()
    MultiAscend.init()

    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      device_num=MultiAscend.get_group_size(),
                                      gradients_mean=True,
                                      full_batch=True)
    set_algo_parameters(elementwise_op_strategy_follow=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CPM inference")
    parser.add_argument('--dataset', type=str, default="", help="dataset path.")
    parser.add_argument('--truth_labels_path', type=str, default="", help="truth_labels path.")
    parser.add_argument('--ckpt_path_doc', type=str, default="", help="checkpoint path doc or checkpoint path.")
    parser.add_argument("--distribute", type=ast.literal_eval, default=False, help='Whether distributed evaluation'
                                                                                   ' with model parallel.')
    parser.add_argument("--has_train_strategy", type=ast.literal_eval, default=False,
                        help='Whether the loaded checkpoints have distributed training strategy.')
    parser.add_argument('--ckpt_partition', type=int, default=1, help="Number of checkpoint partition.")
    args_parse = parser.parse_args()
    if args_parse.distribute:
        set_parallel_env()

    ckpt_file_list_test = None
    if args_parse.has_train_strategy:
        # Get the checkpoint with train strategy.
        train_strategy_list = create_ckpt_file_list(args_parse, train_strategy="train_strategy.ckpt")
        context.set_auto_parallel_context(
            strategy_ckpt_load_file=train_strategy_list[0]
        )
        ckpt_file_list_test = create_ckpt_file_list(args_parse)
        print("Get checkpoint file lists++++", ckpt_file_list_test, flush=True)
    if args_parse.distribute:
        print("Staring evaluating on 2 devices with model parallel.")
        run_eval(args_parse, config_zero_shot_distrubute, ckpt_file_list_test)
    else:
        print("Staring evaluating on 1 device without model parallel.")
        run_eval(args_parse, config_zero_shot_standalone, ckpt_file_list_test)
