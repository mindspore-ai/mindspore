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

from mindspore import context
from mindspore.communication import management as MultiAscend
from mindspore.context import ParallelMode
from mindspore.parallel import set_algo_parameters

from eval import do_eval, create_ckpt_file_list

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num

context.set_context(mode=context.GRAPH_MODE,
                    save_graphs=False,
                    device_target="Ascend",
                    device_id=get_device_id())

def set_parallel_env():
    r"""
    Parallel environment.
    """
    context.reset_auto_parallel_context()
    MultiAscend.init()

    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      device_num=get_device_num(),
                                      gradients_mean=True,
                                      full_batch=True)
    set_algo_parameters(elementwise_op_strategy_follow=True)

def modelarts_pre_process():
    '''modelarts pre process function.'''

@moxing_wrapper(pre_process=modelarts_pre_process)
def run_test():
    '''test cpm network'''
    finetune_test_standalone = config.finetune_test_standalone
    finetune_test_distrubute = config.finetune_test_distrubute
    finetune_dev_standalone = config.finetune_dev_standalone
    finetune_dev_distrubute = config.finetune_dev_distrubute
    if config.distribute:
        set_parallel_env()
        print("Start validation on 2 devices.")
    else:
        print("Start validation on 1 device.")

    config.dataset = config.dev_dataset
    config.dataset_path = config.dev_data_path
    if config.has_train_strategy:
        # Get the checkpoint with train strategy.
        train_strategy_list = create_ckpt_file_list(config, train_strategy="train_strategy.ckpt")
        context.set_auto_parallel_context(
            strategy_ckpt_load_file=train_strategy_list[0]
        )
    # start run in dev dataset.
    ckpt_file_list_dev = None
    if config.has_train_strategy:
        # Get the checkpoint slice.
        ckpt_file_list_dev = create_ckpt_file_list(config, config.ckpt_epoch)
        print("++++ Get sliced checkpoint file, lists: ", ckpt_file_list_dev, flush=True)
    result_i = 0.0
    if config.distribute:
        result_i = do_eval(config, finetune_dev_distrubute, ckpt_file_list_dev)
    else:
        result_i = do_eval(config, finetune_dev_standalone, ckpt_file_list_dev)
    print("+++++ ckpt_epoch=", config.ckpt_epoch, ", dev_dataset Accuracy: ", result_i)
    print("++++ Then we take the model to predict on the test dataset.")
    ckpt_file_list_test = None
    if config.has_train_strategy:
        # Get the best precision checkpoint slice.
        ckpt_file_list_test = create_ckpt_file_list(config, config.ckpt_epoch)

    config.dataset = config.test_dataset
    config.dataset_path = config.test_data_path
    # start run in test dataset.
    result_last = 0.0
    if config.distribute:
        result_last = do_eval(config, finetune_test_distrubute, ckpt_file_list_test)
    else:
        result_last = do_eval(config, finetune_test_standalone, ckpt_file_list_test)
    print("++++ Accuracy on test dataset is: ", result_last)

    # write to file.
    result_path = config.result_path
    if not os.path.exists(result_path):
        with open(result_path, "w") as file:
            file.write("CkptEpcoh  Accuracy_dev  Accuracy_test\n")

    with open(result_path, "a") as file:
        file.write(str(config.ckpt_epoch) + " " + str(result_i) + " " + str(result_last) + "\n")

if __name__ == '__main__':
    run_test()
