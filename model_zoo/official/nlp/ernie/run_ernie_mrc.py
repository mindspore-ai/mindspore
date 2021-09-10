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

'''
Ernie finetune and evaluation script.
'''
import os
import argparse
import collections
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore import log as logger
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.optim import AdamWeightDecay, Adam, Adagrad
from mindspore.common.tensor import Tensor
from mindspore.train.model import Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.context import ParallelMode
from mindspore.communication.management import init
from src.ernie_for_finetune import ErnieMRCCell, ErnieMRC
from src.dataset import create_mrc_dataset
from src.utils import make_directory, LossCallBack, LoadNewestCkpt, ErnieLearningRate
from src.finetune_eval_config import optimizer_cfg, ernie_net_cfg
from src.mrc_get_predictions import write_predictions
from src.mrc_postprocess import mrc_postprocess


_cur_dir = os.getcwd()


def do_train(task_type, dataset=None, network=None, load_checkpoint_path="", save_checkpoint_path="", epoch_num=1):
    """ do train """
    if load_checkpoint_path == "":
        raise ValueError("Pretrain model missed, finetune task must load pretrain model!")
    steps_per_epoch = dataset.get_dataset_size()
    # optimizer
    if optimizer_cfg.optimizer == 'AdamWeightDecay':
        lr_schedule = ErnieLearningRate(learning_rate=optimizer_cfg.AdamWeightDecay.learning_rate,
                                        end_learning_rate=optimizer_cfg.AdamWeightDecay.end_learning_rate,
                                        warmup_steps=int(steps_per_epoch * epoch_num * 0.1),
                                        decay_steps=steps_per_epoch * epoch_num,
                                        power=optimizer_cfg.AdamWeightDecay.power)
        params = network.trainable_params()
        decay_params = list(filter(optimizer_cfg.AdamWeightDecay.decay_filter, params))
        other_params = list(filter(lambda x: not optimizer_cfg.AdamWeightDecay.decay_filter(x), params))
        group_params = [{'params': decay_params, 'weight_decay': optimizer_cfg.AdamWeightDecay.weight_decay},
                        {'params': other_params, 'weight_decay': 0.0}]

        optimizer = AdamWeightDecay(group_params, lr_schedule, eps=optimizer_cfg.AdamWeightDecay.eps)
    elif optimizer_cfg.optimizer == 'Adam':
        optimizer = Adam(network.trainable_params(), learning_rate=optimizer_cfg.Adam.learning_rate)
    elif optimizer_cfg.optimizer == 'Adagrad':
        optimizer = Adagrad(network.trainable_params(), learning_rate=optimizer_cfg.Adagrad.learning_rate)
    # load checkpoint into network
    else:
        raise Exception("Optimizer not supported. support: [AdamWeightDecay, Lamb, Momentum]")

    # load checkpoint into network
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix=task_type,
                                 directory=None if save_checkpoint_path == "" else save_checkpoint_path,
                                 config=ckpt_config)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(network, param_dict)

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2**32, scale_factor=2, scale_window=1000)
    netwithgrads = ErnieMRCCell(network, optimizer=optimizer, scale_update_cell=update_cell)
    model = Model(netwithgrads)
    callbacks = [TimeMonitor(dataset.get_dataset_size()), LossCallBack(dataset.get_dataset_size()), ckpoint_cb]
    model.train(epoch_num, dataset, callbacks=callbacks)

def do_eval(dataset=None, load_checkpoint_path="", eval_batch_size=1):
    """ do eval """
    if load_checkpoint_path == "":
        raise ValueError("Finetune model missed, evaluation task must load finetune model!")
    net = ErnieMRC(ernie_net_cfg, False, 2)
    net.set_train(False)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(net, param_dict)
    model = Model(net)
    output = []
    RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])
    columns_list = ["input_ids", "input_mask", "token_type_id", "unique_id"]
    for data in dataset.create_dict_iterator(num_epochs=1):
        input_data = []
        for i in columns_list:
            input_data.append(data[i])
        input_ids, input_mask, token_type_id, unique_ids = input_data
        start_positions = Tensor([1], mstype.float32)
        end_positions = Tensor([1], mstype.float32)
        logits = model.predict(input_ids, input_mask, token_type_id, start_positions,
                               end_positions, unique_ids)
        ids = logits[0].asnumpy()
        start = logits[1].asnumpy()
        end = logits[2].asnumpy()

        for i, value in enumerate(ids):
            unique_id = int(value)
            start_logits = [float(x) for x in start[i].flat]
            end_logits = [float(x) for x in end[i].flat]
            output.append(RawResult(
                unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits))
    return output

def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="run mrc")
    parser.add_argument("--task_type", type=str, default="drcd", choices=["drcd", "cmrc"],
                        help="Task type, default is drcd")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["Ascend", "GPU"],
                        help="Device type, default is Ascend")
    parser.add_argument("--run_distribute", type=str, default=False, help="Run distribute, default: false.")
    parser.add_argument("--do_train", type=str, default="false", choices=["true", "false"],
                        help="Enable train, default is false")
    parser.add_argument("--do_eval", type=str, default="false", choices=["true", "false"],
                        help="Enable eval, default is false")
    parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default: 1.")
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--rank_id", type=int, default=0, help="Rank id, default: 0.")
    parser.add_argument("--epoch_num", type=int, default=3, help="Epoch number, default is 3.")
    parser.add_argument("--number_labels", type=int, default=3, help="The number of class, default is 3.")
    parser.add_argument("--label_map_config", type=str, default="", help="Label map file path")
    parser.add_argument("--train_data_shuffle", type=str, default="true", choices=["true", "false"],
                        help="Enable train data shuffle, default is true")
    parser.add_argument("--eval_data_shuffle", type=str, default="false", choices=["true", "false"],
                        help="Enable eval data shuffle, default is false")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Train batch size, default is 32")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Eval batch size, default is 1")
    parser.add_argument("--save_finetune_checkpoint_path", type=str, default="", help="Save checkpoint path")
    parser.add_argument("--load_pretrain_checkpoint_path", type=str, default="", help="Load checkpoint file path")
    parser.add_argument("--local_pretrain_checkpoint_path", type=str, default="",
                        help="Local pretrain checkpoint file path")
    parser.add_argument("--load_finetune_checkpoint_path", type=str, default="", help="Load checkpoint file path")
    parser.add_argument("--train_data_file_path", type=str, default="",
                        help="Data path, it is better to use absolute path")
    parser.add_argument("--eval_data_file_path", type=str, default="",
                        help="Data path, it is better to use absolute path")
    parser.add_argument("--eval_json_path", type=str, default="",
                        help="Json data path, it is better to use absolute path")
    parser.add_argument("--vocab_path", type=str, default="", help="vocab file")
    parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
    parser.add_argument('--train_url', type=str, default=None, help='Train output path')
    parser.add_argument('--modelarts', type=str, default='false',
                        help='train on modelarts or not, default is false')
    args_opt = parser.parse_args()

    return args_opt


def run_mrc():
    """run mrc task"""
    args_opt = parse_args()
    epoch_num = args_opt.epoch_num
    load_pretrain_checkpoint_path = args_opt.load_pretrain_checkpoint_path
    save_finetune_checkpoint_path = args_opt.save_finetune_checkpoint_path
    load_finetune_checkpoint_path = args_opt.load_finetune_checkpoint_path
    if args_opt.task_type == 'drcd':
        ernie_net_cfg.seq_length = 512
        optimizer_cfg.AdamWeightDecay.learning_rate = 5e-5
        repeat = 1
    elif args_opt.task_type == 'cmrc':
        ernie_net_cfg.seq_length = 512
        optimizer_cfg.AdamWeightDecay.learning_rate = 3e-5
        repeat = 1
    else:
        raise ValueError("Unsupported task type.")

    if args_opt.run_distribute == 'true':
        if args_opt.device_target == "Ascend":
            rank = args_opt.rank_id
            device_num = args_opt.device_num
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
        elif args_opt.device_target == "GPU":
            init("nccl")
            context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL,
                                              gradients_mean=True)
        else:
            raise ValueError(args_opt.device_target)
    else:
        rank = 0
        device_num = 1

    load_pretrain_checkpoint_path = args_opt.load_pretrain_checkpoint_path
    save_finetune_checkpoint_path = args_opt.save_finetune_checkpoint_path
    load_finetune_checkpoint_path = args_opt.load_finetune_checkpoint_path
    target = args_opt.device_target
    if target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args_opt.device_id)
    elif target == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
        if ernie_net_cfg.compute_type != mstype.float32:
            logger.warning('GPU only support fp32 temporarily, run with fp32.')
            ernie_net_cfg.compute_type = mstype.float32
        if optimizer_cfg.optimizer == 'AdamWeightDecay':
            context.set_context(enable_graph_kernel=True)
    else:
        raise Exception("Target error, GPU or Ascend is supported.")

    if args_opt.do_train.lower() == "true":
        netwithloss = ErnieMRC(ernie_net_cfg, True, num_labels=args_opt.number_labels, dropout_prob=0.1)
        ds = create_mrc_dataset(batch_size=args_opt.train_batch_size,
                                repeat_count=repeat,
                                data_file_path=args_opt.train_data_file_path,
                                rank_size=args_opt.device_num,
                                rank_id=rank,
                                do_shuffle=(args_opt.train_data_shuffle.lower() == "true"),
                                drop_reminder=True)
        print("==============================================================")
        print("processor_name: {}".format(args_opt.device_target))
        print("test_name: ERNIE Finetune Training")
        print("model_name: {}".format("ERNIE + MLP"))
        print("batch_size: {}".format(args_opt.train_batch_size))

        do_train(args_opt.task_type + '-' + str(rank), ds, netwithloss,
                 load_pretrain_checkpoint_path, save_finetune_checkpoint_path, epoch_num)

        if args_opt.do_eval.lower() == "true":
            if save_finetune_checkpoint_path == "":
                load_finetune_checkpoint_dir = _cur_dir
            else:
                load_finetune_checkpoint_dir = make_directory(save_finetune_checkpoint_path)
            load_finetune_checkpoint_path = LoadNewestCkpt(load_finetune_checkpoint_dir,
                                                           ds.get_dataset_size(), epoch_num, args_opt.task_type)

    if args_opt.do_eval.lower() == "true":
        from src.finetune_task_reader import MRCReader
        ds = create_mrc_dataset(batch_size=args_opt.eval_batch_size,
                                repeat_count=1,
                                data_file_path=args_opt.eval_data_file_path,
                                do_shuffle=(args_opt.eval_data_shuffle.lower() == "true"),
                                is_training=False,
                                drop_reminder=False)
        outputs = do_eval(ds, load_finetune_checkpoint_path, args_opt.eval_batch_size)

        reader = MRCReader(
            vocab_path=args_opt.vocab_path,
            max_seq_len=ernie_net_cfg.seq_length,
            do_lower_case=True,
            max_query_len=64,
        )
        eval_examples = reader.read_examples(args_opt.eval_json_path, False)
        eval_features = reader.get_example_features(args_opt.eval_json_path, False)
        all_predictions = write_predictions(eval_examples, eval_features, outputs, 20, 100, True)
        mrc_postprocess(args_opt.eval_json_path, all_predictions)

if __name__ == "__main__":
    run_mrc()
