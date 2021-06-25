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
"""Eval."""
import os
import ast
import argparse
import json
import numpy as np

from mindspore import context, load_distributed_checkpoint
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore.train.model import Model
import mindspore.common.dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication import management as MultiAscend
from mindspore.context import ParallelMode
from mindspore.parallel import set_algo_parameters

from src.cpm import CPMModel
from src.cpm_train import VirtualDatasetOneInputCell
from src.cpm_loss import Cross_entropy_eval
from src.config import finetune_test_distrubute, finetune_test_standalone
from train import load_dataset

device_id = int(os.getenv("DEVICE_ID"))
rank_size = os.getenv('RANK_SIZE')
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
        self.loss_net = Cross_entropy_eval(batch_size=self.batch_size,
                                           seq_length=self.seq_length,
                                           vocab_size=self.vocab_size,
                                           config=config)

    def construct(self, input_ids, position_ids, attention_mask, loss_mask):
        logits = self.network(input_ids, position_ids, attention_mask)
        loss = self.loss_net(logits, loss_mask)
        return loss


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
    with open(args.data_path, "r") as f:
        # cand_ids, data
        cand_ids, _ = json.load(f)
    print("++++ cand_ids: ", cand_ids)

    if args.distribute:
        dataset = load_dataset(args.dataset, config_eval.batch_size,
                               rank_size=MultiAscend.get_group_size(),
                               rank_id=MultiAscend.get_rank(),
                               drop_remainder=False,
                               is_training=False,
                               shuffle=False)
    else:
        dataset = load_dataset(args.dataset,
                               config_eval.batch_size,
                               drop_remainder=False,
                               is_training=False,
                               shuffle=False)

    cpm_model = CPM_LAYER(config_eval)

    if args.distribute:
        cpm_model = VirtualDatasetOneInputCell(cpm_model)
    params = cpm_model.trainable_params()
    print("+++++++current network parameter+++++")
    for pas in params:
        print(pas.name)
    print("++++++++++++")
    if not args.has_train_strategy:
        # load the checkpoint without train strategy.
        weights = load_checkpoint(args.ckpt_path_doc)
        can_be_loaded = {}
        print("+++++++loading weights+++++")
        for name, _ in weights.items():
            print('oldname:           ' + name)
            if 'cpm_model.' not in name:
                can_be_loaded['cpm_model.' + name] = weights[name]

                print('newname: cpm_model.' + name)
            else:
                can_be_loaded[name] = weights[name]
        print("+++++++loaded weights+++++")
        load_param_into_net(cpm_model, parameter_dict=can_be_loaded)

    infer_net = CPMForInfer(network=cpm_model,
                            batch_size=config_eval.batch_size,
                            seq_length=config_eval.seq_length,
                            vocab_size=config_eval.vocab_size,
                            config=config_eval)

    model = Model(infer_net)

    if args.has_train_strategy and not args.distribute:
        # load sliced checkpoint with train strategy, but will run standalone inference without model parallel.
        load_distributed_checkpoint(infer_net, ckpt_file_list, None)

    if args.has_train_strategy and args.distribute:
        # load sliced checkpoint with train strategy, will run distribute inference with model parallel.
        fake_input_ids = Tensor(np.ones((config_eval.batch_size, config_eval.seq_length)), mstype.int64)
        fake_position_ids = Tensor(np.random.randint(0, 10, [config_eval.batch_size, config_eval.seq_length]),
                                   mstype.int64)
        fake_attention_mask = Tensor(
            np.random.randn(config_eval.batch_size, config_eval.seq_length, config_eval.seq_length), mstype.float16)
        fake_loss_mask = Tensor(np.random.randn(config_eval.batch_size, config_eval.seq_length), mstype.float16)
        predict_layout = model.infer_predict_layout(fake_input_ids,
                                                    fake_position_ids,
                                                    fake_attention_mask,
                                                    fake_loss_mask)
        print("Loaded sliced checkpoint, will run distribute inference with model parallel.", flush=True)
        load_distributed_checkpoint(infer_net, ckpt_file_list, predict_layout)

    all_losses = []
    truth_labels = []

    steps_per_epoch = dataset.get_dataset_size()
    print("++++++Dataset size", steps_per_epoch, flush=True)

    for batch in dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
        print("++++ start")
        ms_truth = batch['truth']

        input_ids = Tensor(batch['input_ids'], mstype.int64)
        position_ids = Tensor(batch['position_ids'], mstype.int64)
        attention_mask = Tensor(batch['attention_mask'], mstype.float16)
        loss_mask = Tensor(batch['loss_mask'], mstype.float16)

        pred_id_tensor = model.predict(input_ids, position_ids, attention_mask, loss_mask)
        # numpy do it.
        pred_id_np = pred_id_tensor.asnumpy()
        pred_id_np = pred_id_np[:, cand_ids]
        pred_id = pred_id_np.argmax(axis=-1)
        print("++++ pred_id_np: ", pred_id_np)
        print("++++ ms_truth: ", ms_truth)

        all_losses.append(pred_id)
        truth_labels.append(ms_truth)

    all_losses = np.stack(all_losses).reshape(-1)
    truth_labels = np.stack(truth_labels).reshape(-1)
    print("++++ all_losses= \n", all_losses)
    print("++++ truthlabel= \n", truth_labels)
    result = sum([int(p == l) for p, l in zip(all_losses, truth_labels)]) / len(truth_labels)
    print("RESULT: ", result)
    return result


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


def find_file(path, qianzhui):
    r'''
    Find the file address according to the prefix.
    '''
    result_addr = None
    for i, _, k in os.walk(path):
        for file in k:
            if file.startswith(qianzhui):
                result_addr = os.path.join(i, file)
                break
    return result_addr


def create_ckpt_file_list(args, max_index=None, train_strategy=None, steps_per_epoch=4509):
    """user-defined ckpt file list"""
    ckpt_file_list = []
    # train_strategy
    if train_strategy is not None:
        true_path = find_file(args.ckpt_path_doc, train_strategy)
        if true_path is not None:
            ckpt_file_list.append(true_path)
        else:
            raise Exception("+++ ckpt not found!!! +++")
        return ckpt_file_list

    # order in rank_id
    for i in range(0, args.ckpt_partition):
        path_name = "cpm_rank_" + str(i) + "-"
        if max_index is not None:
            path_name = path_name + str(max_index) + "_"
        true_path = find_file(args.ckpt_path_doc, path_name)
        if true_path is not None:
            ckpt_file_list.append(true_path)
        else:
            path_name = "cpm_rank_" + str(i) + "-" + str(max_index * steps_per_epoch)
            true_path = find_file(args.ckpt_path_doc, path_name)
            if true_path is not None:
                ckpt_file_list.append(true_path)
            else:
                raise Exception("+++ ckpt not found!!! +++")
    return ckpt_file_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CPM inference")
    parser.add_argument('--dataset', type=str, default="", help="dataset path.")
    parser.add_argument("--data_path", type=str, default="/disk0/dataset/finetune_dataset/test.json",
                        help='test_json path.')
    parser.add_argument('--ckpt_path_doc', type=str, default="", help="Checkpoint path document.")
    parser.add_argument('--ckpt_partition', type=int, default=8, help="Number of checkpoint partition.")
    parser.add_argument("--distribute", type=ast.literal_eval, default=False,
                        help='Distribute evaluating with model parallel.')
    parser.add_argument("--has_train_strategy", type=ast.literal_eval, default=True,
                        help='Model has distributed training strategy.')
    args_eval = parser.parse_args()
    if args_eval.distribute:
        set_parallel_env()

    ckpt_file_list_test = None
    if args_eval.has_train_strategy:
        # Get the checkpoint with train strategy.
        train_strategy_list = create_ckpt_file_list(args_eval, train_strategy="train_strategy.ckpt")
        context.set_auto_parallel_context(
            strategy_ckpt_load_file=train_strategy_list[0]
        )
        ckpt_file_list_test = create_ckpt_file_list(args_eval)
        print("++++ Get sliced checkpoint file, lists: ", ckpt_file_list_test, flush=True)

    result_accuracy = 0.0
    if args_eval.distribute:
        print("Start validation on 2 devices with model parallel.")
        result_accuracy = run_eval(args_eval, finetune_test_distrubute, ckpt_file_list_test)
    else:
        print("Start validation on 1 device without model parallel.")
        result_accuracy = run_eval(args_eval, finetune_test_standalone, ckpt_file_list_test)

    print("++++ Accuracy=", result_accuracy)
