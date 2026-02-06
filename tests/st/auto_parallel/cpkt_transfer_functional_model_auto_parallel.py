# Copyright 2025 Huawei Technologies Co., Ltd
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

"""test ckpt trans cases"""
import os
import math
import numpy as np

import mindspore as ms
import mindspore.dataset as ds
from mindspore import context, nn, ops, jit
from mindspore.communication import init
from mindspore.communication.management import get_rank, get_group_size
from mindspore.common import Parameter, set_seed
from mindspore.common.initializer import initializer, HeUniform
from mindspore.nn.utils import no_init_parameters
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, LossMonitor, save_checkpoint, \
    load_checkpoint, load_param_into_net
from mindspore.parallel import load_distributed_checkpoint, rank_list_for_convert, convert_checkpoint_by_rank, \
    load_segmented_checkpoints
from tests.st.auto_parallel.utils._utils import set_parallel_mode, clean_all_ckpt_files, compare_params, \
    parallel_mode_get_ckpt_path_with_strategy, clear_files_in_directory

context.set_context(mode=context.GRAPH_MODE)
init()

# dataset
step_per_epoch = 4
var_single_batch_size = 16
var_in_dim = 32
var_hidden_dim = 8
var_out_dim = 16


def get_dataset(*input_data):
    def generate():
        for _ in range(step_per_epoch):
            yield input_data

    return generate


# define net
class Net(nn.Cell):
    """define net"""

    def __init__(self, in_dim, hidden_dim, out_dim, strategy=None):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        weight_init = HeUniform(math.sqrt(5))
        self.weight = Parameter(initializer(weight_init, [self.in_dim, self.hidden_dim]), "w")
        self.weight2 = Parameter(initializer(weight_init, [self.hidden_dim, self.out_dim]), "w2")

        self.matmul = ops.MatMul()
        if strategy is not None:
            self.matmul.shard(strategy)
        self.relu = ops.ReLU()
        self.matmul2 = ops.MatMul()

    def construct(self, x):
        out = self.matmul(x, self.weight)
        out = self.relu(out)
        out = self.matmul2(out, self.weight2)
        return out


def graph_model_train(parallel_config=None, ckpt_dirs=None, ckpt_prefix=None):
    # dataset
    set_seed(1)
    input_data = ms.Tensor(np.random.rand(var_single_batch_size, var_in_dim).astype(np.float32))
    label_data = ms.Tensor(np.random.rand(var_single_batch_size, var_out_dim).astype(np.float32))
    fake_dataset = get_dataset(input_data, label_data)
    dataset = ds.GeneratorDataset(fake_dataset, ["input", "label"])

    # net
    strategy = ((2, 4), (4, 1))
    if parallel_config is None:
        net = Net(var_in_dim, var_hidden_dim, var_out_dim, strategy)
        net_opt = nn.Momentum(net.trainable_params(), 0.01, 0.9)
    else:
        with no_init_parameters():
            net = Net(var_in_dim, var_hidden_dim, var_out_dim, strategy)
            net_opt = nn.Momentum(net.trainable_params(), 0.01, 0.9)

    # creat model
    net_loss = nn.SoftmaxCrossEntropyWithLogits(reduction="mean")
    net = set_parallel_mode(net, parallel_config)
    model = Model(network=net, loss_fn=net_loss, optimizer=net_opt)

    # model train
    epoch = 2
    clean_all_ckpt_files(ckpt_dirs)
    ckpt_config = CheckpointConfig(keep_checkpoint_max=1, integrated_save=False)
    loss_cb = LossMonitor(1)
    ckpt_callback = ModelCheckpoint(prefix=ckpt_prefix, directory=ckpt_dirs, config=ckpt_config)
    model.train(epoch=epoch, train_dataset=dataset, callbacks=[ckpt_callback, loss_cb], dataset_sink_mode=False)


def graph_model_predict(checkpoint_filenames=None, train_strategy_file=None, parallel_config=None):
    # predict dataset
    set_seed(1)
    predict_data = ms.Tensor(np.random.rand(var_single_batch_size, var_in_dim).astype(np.float32))

    # define predict net with delay init
    if parallel_config is None:
        net = Net(var_in_dim, var_hidden_dim, var_out_dim, strategy=None)
    else:
        with no_init_parameters():
            net = Net(var_in_dim, var_hidden_dim, var_out_dim, strategy=None)

    # set parallel mode
    net = set_parallel_mode(net, parallel_config)

    # tansfer strategy
    model = Model(net)
    predict_strategy = model.infer_predict_layout(predict_data)
    ms.mint.distributed.barrier()
    load_distributed_checkpoint(net, checkpoint_filenames, predict_strategy, train_strategy_file)
    predict_result = model.predict(predict_data)
    return predict_result


# train and predict by model, set parallel by context
def cpkt_transfer_model_context():
    exec_path = os.path.dirname(os.path.realpath(__file__))
    cur_dir = os.path.join(exec_path, "test_cpkt_transfer/model_context")
    ir_graph_path = f'{cur_dir}/graphs'
    context.set_context(save_graphs=True, save_graphs_path=ir_graph_path)
    clear_files_in_directory(f"{ir_graph_path}/rank_{get_rank()}")

    # train
    context.reset_auto_parallel_context()
    strategy_ckpt_config = {"save_file": f"{cur_dir}/train_strategy.ckpt",
                            "only_trainable_params": True}
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8,
                                      strategy_ckpt_config=strategy_ckpt_config)
    ckpt_dirs = f"{cur_dir}/rank_{get_rank()}"
    ckpt_prefix = "ckpt_model_context"

    graph_model_train(parallel_config=None, ckpt_dirs=ckpt_dirs, ckpt_prefix=ckpt_prefix)
    context.reset_auto_parallel_context()

    # find checkpoint file with train strategy
    parallel_mode_get_ckpt_path_with_strategy(strategy_ckpt_config["save_file"], ckpt_dirs)

    # predict
    train_strategy_file = strategy_ckpt_config["save_file"]
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8)
    checkpoint_filenames = [f"{cur_dir}/rank_{i}/{ckpt_prefix}-2_4.ckpt" for i in range(get_group_size())]
    my_predict = graph_model_predict(checkpoint_filenames, train_strategy_file)

    context.reset_auto_parallel_context()
    return my_predict


# train and predict by model, set parallel by auto_parallel
def cpkt_transfer_model_auto_parallel():
    exec_path = os.path.dirname(os.path.realpath(__file__))
    cur_dir = os.path.join(exec_path, "test_cpkt_transfer/model_auto_parallel")
    ir_graph_path = f'{cur_dir}/graphs'
    context.set_context(save_graphs=True, save_graphs_path=ir_graph_path)
    clear_files_in_directory(f"{ir_graph_path}/rank_{get_rank()}")

    # train
    stra_ckpt_file = f"{cur_dir}/train_strategy.ckpt"
    train_parallel_config = {"parallel_mode": "semi_auto", "save_strategy_file": stra_ckpt_file}
    ckpt_dirs = f"{cur_dir}/rank_{get_rank()}_ckpt"
    ckpt_prefix = "ckpt_model_auto_parallel"
    graph_model_train(parallel_config=train_parallel_config, ckpt_dirs=ckpt_dirs, ckpt_prefix=ckpt_prefix)

    # find checkpint file with train strategy
    parallel_mode_get_ckpt_path_with_strategy(stra_ckpt_file, ckpt_dirs)

    # predict
    train_strategy_file = stra_ckpt_file
    predict_parallel_config = {"parallel_mode": "semi_auto"}
    checkpoint_filenames = [f"{cur_dir}/rank_{i}_ckpt/{ckpt_prefix}-2_4.ckpt" for i in range(get_group_size())]
    my_predict = graph_model_predict(checkpoint_filenames, train_strategy_file, predict_parallel_config)
    return my_predict


def graph_func_train(parallel=True, parallel_config=None, cpkt_file=None):
    # dataset
    set_seed(1)
    input_data = ms.Tensor(np.random.rand(var_single_batch_size, var_in_dim).astype(np.float32))
    label_data = ms.Tensor(np.random.rand(var_single_batch_size, var_out_dim).astype(np.float32))
    fake_dataset = get_dataset(input_data, label_data)
    dataset = ds.GeneratorDataset(fake_dataset, ["input", "label"])

    # define shard
    strategy = None
    if parallel:
        strategy = ((2, 4), (4, 1))

    # define train net and opt with delay init
    with no_init_parameters():
        net = Net(var_in_dim, var_hidden_dim, var_out_dim, strategy)
        opt = nn.Momentum(params=net.trainable_params(), learning_rate=0.01, momentum=0.9)

    # define loss
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(reduction="mean")

    # 1. define forward function
    def net_forward(x, y):
        out = net(x)
        loss = loss_fn(out, y)
        return loss

    # 2. Get gradient function
    grad_net = ops.value_and_grad(net_forward, grad_position=None, weights=net.trainable_params())

    # 3. Define function of one-step training
    @jit
    def train_one_step(inputs, target):
        loss, grads = grad_net(inputs, target)
        opt(grads)
        return loss

    # 4. set parallel mode
    parallel_net = set_parallel_mode(train_one_step, parallel_config)

    # 5. compile and init data, train net
    for _ in range(2):
        for input_x, label in dataset:
            loss = parallel_net(input_x, label)
            print("loss", loss, flush=True)

    # 6. save_checkpoint of net params
    ckpt_dir = os.path.dirname(cpkt_file)
    clean_all_ckpt_files(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(cpkt_file, 'w', encoding="utf-8") as file:
        file.write("")
    save_checkpoint(net, cpkt_file, integrated_save=True)
    ms.mint.distributed.barrier()


def graph_func_predict_net(parallel_config=None):
    with no_init_parameters():
        net = Net(var_in_dim, var_hidden_dim, var_out_dim, strategy=None)

    predict_net = set_parallel_mode(net, parallel_config)
    return predict_net


def graph_func_predict_convert_checkpoint_by_rank(src_strategy_file, dst_strategy_file, src_checkpoints_files,
                                                  dst_checkpoints_files):
    rank_id = get_rank()
    ckpt_file = dst_checkpoints_files[rank_id]
    ckpt_dir = os.path.dirname(ckpt_file)
    os.makedirs(ckpt_dir, exist_ok=True)
    clean_all_ckpt_files(ckpt_dir)
    rank_list = rank_list_for_convert(rank_id, src_strategy_file, dst_strategy_file)
    checkpoint_files_map = {}
    for rank in rank_list:
        checkpoint_files_map[rank] = src_checkpoints_files[rank]
    save_checkpoint_file_name = dst_checkpoints_files[rank_id]
    convert_checkpoint_by_rank(rank_id, checkpoint_files_map, save_checkpoint_file_name, src_strategy_file,
                               dst_strategy_file)


# functional train and predict, set parallel by context
def cpkt_transfer_func_auto_parallel():
    exec_path = os.path.dirname(os.path.realpath(__file__))
    cur_dir = os.path.join(exec_path, "test_cpkt_transfer/functional_auto_parallel")
    ir_graph_path = f'{cur_dir}/graphs'
    context.set_context(save_graphs=True, save_graphs_path=ir_graph_path)
    clear_files_in_directory(f"{ir_graph_path}/rank_{get_rank()}")

    # parallel train
    train_stra_ckpt_file = f"{cur_dir}/train_strategy.ckpt"
    train_parallel_config = {"parallel_mode": "semi_auto", "data_strategy": "data_parallel",
                             "save_strategy_file": train_stra_ckpt_file}

    train_ckpt_dirs = f"{cur_dir}/src_checkpoints_dir/rank_{get_rank()}"
    src_ckpt_file = f"{train_ckpt_dirs}/src_checkpoint_{get_rank()}.ckpt"
    graph_func_train(parallel=True, parallel_config=train_parallel_config, cpkt_file=src_ckpt_file)

    # find checkpoint file with train strategy
    parallel_mode_get_ckpt_path_with_strategy(train_stra_ckpt_file, train_ckpt_dirs)

    # predict dataset
    set_seed(1)
    predict_data = ms.Tensor(np.random.rand(var_single_batch_size, var_in_dim).astype(np.float32))

    # parallel predict net
    predict_stra_ckpt_file = f"{cur_dir}/predict_strategy.ckpt"
    predict_parallel_config = {"parallel_mode": "semi_auto", "data_strategy": "data_parallel",
                               "save_strategy_file": predict_stra_ckpt_file}
    predict_net = graph_func_predict_net(predict_parallel_config)

    # transfer1: load checkpoint files and predict
    src_checkpoint_files = [f"{cur_dir}/src_checkpoints_dir/rank_{i}/src_checkpoint_{i}.ckpt" for i in
                            range(get_group_size())]
    train_strategy_file = train_stra_ckpt_file
    predict_strategy = predict_net.parameter_layout_dict
    ms.mint.distributed.barrier()
    load_distributed_checkpoint(network=predict_net, checkpoint_filenames=src_checkpoint_files,
                                predict_strategy=predict_strategy, train_strategy_filename=train_strategy_file)
    predict_result_load_distributed_checkpoint = predict_net(predict_data)

    # transfer2: rank_list_for_convert, convert_checkpoint_by_rank
    dst_checkpoint_files = [f"{cur_dir}/dst_checkpoints_dir/rank_{i}/dst_checkpoint_{i}.ckpt" for i in
                            range(get_group_size())]
    graph_func_predict_convert_checkpoint_by_rank(train_stra_ckpt_file, predict_stra_ckpt_file, src_checkpoint_files,
                                                  dst_checkpoint_files)
    param_dict = load_checkpoint(dst_checkpoint_files[get_rank()], net=predict_net)
    param_not_load, _ = load_param_into_net(net=predict_net, parameter_dict=param_dict)
    print("param_not_load in transform_checkpoints", param_not_load)
    predict_result_transform_checkpoints = predict_net(predict_data)

    # transfer3: load_segmented_checkpoints
    ckpt_file = dst_checkpoint_files[get_rank()]
    ckpt_dir = os.path.dirname(ckpt_file)
    param_dict_new = load_segmented_checkpoints(ckpt_file_dir=ckpt_dir, net=predict_net)
    param_not_load, _ = load_param_into_net(net=predict_net, parameter_dict=param_dict_new)
    print("param_not_load in load_segmented_checkpoints", param_not_load)
    predict_result_load_segmented_checkpoints = predict_net(predict_data)

    #  predict result
    res = {"load_distributed_checkpoint": predict_result_load_distributed_checkpoint,
           "convert_checkpoints": predict_result_transform_checkpoints,
           "load_segmented_checkpoints": predict_result_load_segmented_checkpoints}
    return res


def test_parallel_functional_train_predict():
    """
    Feature: test save_checkpoints, load_checkpoint and convert checkpoints.
    Description:
        1. define model train and predict, train net and save checkpoints with model, load_distributed_checkpoint to
        predict net.
        2. case 1 set parallel by context, train and predict net with model.
        3. case 2 set parallel by auto_parallel interface, train and predict net with model.
        3. define functional train and predict, set parallel by auto_parallel.
        4. case 3, load_distributed_checkpoint for predict net.
        5. case 4, convert_checkpoint_by_rank, load_checkpoint and load_param_into_net to predict net.
        6. case 5, load_segmented_checkpoints and load_param_into_net to predict net.
        7. compare accuracy of predict result of case 2 ~ case 5 with case 1.
    Expectation:
        1.train and predict ok
        2.the prediction accuracy meets the requirements.
    """
    # model train and predict, case 1 and case 2
    context_model_predict_result = cpkt_transfer_model_context()
    print(f"context_model_predict_result {context_model_predict_result}")

    auto_parallel_model_predict_result = cpkt_transfer_model_auto_parallel()
    print(f"auto_parallel_model_predict_result {auto_parallel_model_predict_result}")

    # functional train and predict, case 3, case4 and case 5
    function_auto_predict_result_dict = cpkt_transfer_func_auto_parallel()
    function_load_distributed_checkpoint = function_auto_predict_result_dict.get("load_distributed_checkpoint")
    function_convert_checkpoints = function_auto_predict_result_dict.get("convert_checkpoints")
    function_load_segmented_checkpoints = function_auto_predict_result_dict.get("load_segmented_checkpoints")
    print("function_predict_result_load_distributed_checkpoint", function_load_distributed_checkpoint)
    print("function_predict_result_convert_checkpoints", function_convert_checkpoints)
    print("function_predict_result_load_segmented_checkpoints", function_load_segmented_checkpoints)

    # accuracy
    compare_params(function_load_distributed_checkpoint, context_model_predict_result)
    compare_params(function_convert_checkpoints, context_model_predict_result)
    compare_params(function_load_segmented_checkpoints, context_model_predict_result)
    compare_params(auto_parallel_model_predict_result, context_model_predict_result)
