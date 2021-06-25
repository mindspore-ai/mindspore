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
"""Test."""
import os
import ast
import argparse

from mindspore import context
from mindspore.communication import management as MultiAscend
from mindspore.context import ParallelMode
from mindspore.parallel import set_algo_parameters

from src.config import finetune_test_standalone, finetune_test_distrubute, \
    finetune_dev_distrubute, finetune_dev_standalone
from eval import run_eval, create_ckpt_file_list

device_id = int(os.getenv("DEVICE_ID"))
rank_size = os.getenv('RANK_SIZE')
context.set_context(mode=context.GRAPH_MODE,
                    save_graphs=False,
                    device_target="Ascend",
                    device_id=device_id)


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
    parser.add_argument('--dev_dataset', type=str, default="", help="dev_dataset path.")
    parser.add_argument("--dev_data_path", type=str, default="/disk0/dataset/finetune_dataset/dev.json",
                        help='dev_json path.')
    parser.add_argument('--test_dataset', type=str, default="", help="test_dataset path.")
    parser.add_argument("--test_data_path", type=str, default="/disk0/dataset/finetune_dataset/test.json",
                        help='test_json path.')
    parser.add_argument('--ckpt_path_doc', type=str, default="", help="checkpoint path document.")
    parser.add_argument('--ckpt_partition', type=int, default=8, help="Number of checkpoint partition.")
    parser.add_argument("--distribute", type=ast.literal_eval, default=False,
                        help='Whether distributed evaluation with model parallel.')
    parser.add_argument("--has_train_strategy", type=ast.literal_eval, default=True,
                        help='Whether the loaded checkpoints have distributed training strategy.')
    parser.add_argument("--result_path", type=str, default="/home/result.txt",
                        help='Text save address.')
    parser.add_argument("--ckpt_epoch", type=int, default=4,
                        help='The number of checkpoint epochs.')
    args_eval = parser.parse_args()

    if args_eval.distribute:
        set_parallel_env()
        print("Start validation on 2 devices.")
    else:
        print("Start validation on 1 device.")

    args_eval.dataset = args_eval.dev_dataset
    args_eval.data_path = args_eval.dev_data_path
    if args_eval.has_train_strategy:
        # Get the checkpoint with train strategy.
        train_strategy_list = create_ckpt_file_list(args_eval, train_strategy="train_strategy.ckpt")
        context.set_auto_parallel_context(
            strategy_ckpt_load_file=train_strategy_list[0]
        )
    # start run in dev dataset.
    ckpt_file_list_dev = None
    if args_eval.has_train_strategy:
        # Get the checkpoint slice.
        ckpt_file_list_dev = create_ckpt_file_list(args_eval, args_eval.ckpt_epoch)
        print("++++ Get sliced checkpoint file, lists: ", ckpt_file_list_dev, flush=True)
    result_i = 0.0
    if args_eval.distribute:
        result_i = run_eval(args_eval, finetune_dev_distrubute, ckpt_file_list_dev)
    else:
        result_i = run_eval(args_eval, finetune_dev_standalone, ckpt_file_list_dev)
    print("+++++ ckpt_epoch=", args_eval.ckpt_epoch, ", dev_dataset Accuracy: ", result_i)
    print("++++ Then we take the model to predict on the test dataset.")
    ckpt_file_list_test = None
    if args_eval.has_train_strategy:
        # Get the best precision checkpoint slice.
        ckpt_file_list_test = create_ckpt_file_list(args_eval, args_eval.ckpt_epoch)

    args_eval.dataset = args_eval.test_dataset
    args_eval.data_path = args_eval.test_data_path
    # start run in test dataset.
    result_last = 0.0
    if args_eval.distribute:
        result_last = run_eval(args_eval, finetune_test_distrubute, ckpt_file_list_test)
    else:
        result_last = run_eval(args_eval, finetune_test_standalone, ckpt_file_list_test)
    print("++++ Accuracy on test dataset is: ", result_last)

    # write to file.
    result_path = args_eval.result_path
    if not os.path.exists(result_path):
        with open(result_path, "w") as file:
            file.write("CkptEpcoh  Accuracy_dev  Accuracy_test\n")

    with open(result_path, "a") as file:
        file.write(str(args_eval.ckpt_epoch) + " " + str(result_i) + " " + str(result_last) + "\n")
