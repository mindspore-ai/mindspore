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

"""train CTPN and get checkpoint files."""
import os
import ast
import operator
import mindspore.common.dtype as mstype
from mindspore import context, Tensor
from mindspore.communication.management import init
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn import Momentum
from mindspore.common import set_seed
from src.ctpn import CTPN
from src.dataset import create_ctpn_dataset
from src.lr_schedule import dynamic_lr
from src.network_define import LossCallBack, LossNet, WithLossCell, TrainOneStepCell
from src.eval_utils import eval_for_ctpn, get_eval_result
from src.eval_callback import EvalCallBack
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_num, get_device_id, get_rank_id


set_seed(1)


context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=get_device_id(), save_graphs=True)


binOps = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod
}


def arithmeticeval(s):
    node = ast.parse(s, mode='eval')

    def _eval(node):
        if isinstance(node, ast.BinOp):
            return binOps[type(node.op)](_eval(node.left), _eval(node.right))

        if isinstance(node, ast.Num):
            return node.n

        if isinstance(node, ast.Expression):
            return _eval(node.body)

        raise Exception('unsupported type{}'.format(node))
    return _eval(node.body)


def apply_eval(eval_param):
    network = eval_param["eval_network"]
    eval_ds = eval_param["eval_dataset"]
    eval_image_path = eval_param["eval_image_path"]
    eval_for_ctpn(network, eval_ds, eval_image_path)
    hmean = get_eval_result()
    return hmean


def modelarts_pre_process():
    pass


@moxing_wrapper(pre_process=modelarts_pre_process)
def train():
    config.feature_shapes = [config.img_height // 16, config.img_width // 16]
    config.num_bboxes = (config.img_height // 16) * (config.img_width // 16) * config.num_anchors
    config.num_step = config.img_width // 16
    config.rnn_batch_size = config.img_height // 16
    config.weight_decay = arithmeticeval(config.weight_decay)

    if config.run_distribute:
        rank = get_rank_id()
        device_num = get_device_num()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        init()
    else:
        rank = 0
        device_num = 1
    if config.task_type == "Pretraining":
        print("Start to do pretraining")
        mindrecord_file = config.pretraining_dataset_file
        config.base_lr = config.pre_base_lr
        config.warmup_step = config.pre_warmup_step
        config.warmup_ratio = arithmeticeval(config.pre_warmup_ratio)
        config.total_epoch = config.pre_total_epoch
    else:
        print("Start to do finetune")
        mindrecord_file = config.finetune_dataset_file
        config.base_lr = config.fine_base_lr
        config.warmup_step = config.fine_warmup_step
        config.warmup_ratio = arithmeticeval(config.fine_warmup_ratio)
        config.total_epoch = config.fine_total_epoch

    print("CHECKING MINDRECORD FILES DONE!")

    # loss_scale = float(config.loss_scale)

    # When create MindDataset, using the fitst mindrecord file, such as ctpn_pretrain.mindrecord0.
    dataset = create_ctpn_dataset(mindrecord_file, repeat_num=1, \
        batch_size=config.batch_size, device_num=device_num, rank_id=rank)
    dataset_size = dataset.get_dataset_size()
    net = CTPN(config=config, batch_size=config.batch_size)
    net = net.set_train()

    load_path = config.pre_trained
    if config.task_type == "Pretraining":
        print("load backbone vgg16 ckpt {}".format(config.pre_trained))
        param_dict = load_checkpoint(load_path)
        for item in list(param_dict.keys()):
            if not item.startswith('vgg16_feature_extractor'):
                param_dict.pop(item)
        load_param_into_net(net, param_dict)
    else:
        if load_path != "":
            print("load pretrain ckpt {}".format(config.pre_trained))
            param_dict = load_checkpoint(load_path)
            load_param_into_net(net, param_dict)
    loss = LossNet()
    lr = Tensor(dynamic_lr(config, dataset_size), mstype.float32)
    opt = Momentum(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum,\
        weight_decay=config.weight_decay, loss_scale=config.loss_scale)
    net_with_loss = WithLossCell(net, loss)
    if config.run_distribute:
        net_with_grads = TrainOneStepCell(net_with_loss, opt, sens=config.loss_scale, reduce_flag=True, \
            mean=True, degree=device_num)
    else:
        net_with_grads = TrainOneStepCell(net_with_loss, opt, sens=config.loss_scale)

    time_cb = TimeMonitor(data_size=dataset_size)
    loss_cb = LossCallBack(rank_id=rank)
    cb = [time_cb, loss_cb]
    save_checkpoint_path = os.path.join(config.save_checkpoint_path, "ckpt_" + str(rank) + "/")
    if config.save_checkpoint:
        ckptconfig = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs*dataset_size,
                                      keep_checkpoint_max=config.keep_checkpoint_max)
        ckpoint_cb = ModelCheckpoint(prefix='ctpn', directory=save_checkpoint_path, config=ckptconfig)
        cb += [ckpoint_cb]
    if config.run_eval:
        if config.eval_dataset_path is None or (not os.path.isfile(config.eval_dataset_path)):
            raise ValueError("{} is not a existing path.".format(config.eval_dataset_path))
        if config.eval_image_path is None or (not os.path.isdir(config.eval_image_path)):
            raise ValueError("{} is not a existing path.".format(config.eval_image_path))
        eval_dataset = create_ctpn_dataset(config.eval_dataset_path, \
            batch_size=config.batch_size, repeat_num=1, is_training=False)
        eval_net = net
        eval_param_dict = {"eval_network": eval_net, "eval_dataset": eval_dataset, \
            "eval_image_path": config.eval_image_path}
        eval_cb = EvalCallBack(apply_eval, eval_param_dict, interval=config.eval_interval,
                               eval_start_epoch=config.eval_start_epoch, save_best_ckpt=True,
                               ckpt_directory=save_checkpoint_path, besk_ckpt_name="best_acc.ckpt",
                               metrics_name="hmean")
        cb += [eval_cb]
    model = Model(net_with_grads)
    model.train(config.total_epoch, dataset, callbacks=cb, dataset_sink_mode=True)


if __name__ == '__main__':
    train()
