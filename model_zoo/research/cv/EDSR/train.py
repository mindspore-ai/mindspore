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
"""
#################train EDSR example on DIV2K########################
"""
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor, Callback
from mindspore.train.model import Model
from mindspore.common import set_seed

from src.metric import PSNR
from src.utils import init_env, init_dataset, init_net, modelarts_pre_process, do_eval
from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_rank_id, get_device_num


set_seed(2021)


def lr_steps_edsr(lr, milestones, gamma, epoch_size, steps_per_epoch, last_epoch=None):
    lr_each_step = []
    step_begin_epoch = [0] + milestones[:-1]
    step_end_epoch = milestones[1:] + [epoch_size]
    for begin, end in zip(step_begin_epoch, step_end_epoch):
        lr_each_step += [lr] * (end - begin) * steps_per_epoch
        lr *= gamma
    if last_epoch is not None:
        lr_each_step = lr_each_step[last_epoch * steps_per_epoch:]
    return np.array(lr_each_step).astype(np.float32)


def init_opt(cfg, net):
    """
    init opt to train edsr
    """
    lr = lr_steps_edsr(lr=cfg.learning_rate, milestones=cfg.milestones, gamma=cfg.gamma,
                       epoch_size=cfg.epoch_size, steps_per_epoch=cfg.steps_per_epoch, last_epoch=None)
    loss_scale = 1.0 if cfg.amp_level == "O0" else cfg.loss_scale
    if cfg.opt_type == "Adam":
        opt = nn.Adam(params=filter(lambda x: x.requires_grad, net.get_parameters()),
                      learning_rate=Tensor(lr),
                      weight_decay=cfg.weight_decay,
                      loss_scale=loss_scale)
    elif cfg.opt_type == "SGD":
        opt = nn.SGD(params=filter(lambda x: x.requires_grad, net.get_parameters()),
                     learning_rate=Tensor(lr),
                     weight_decay=cfg.weight_decay,
                     momentum=cfg.momentum,
                     dampening=cfg.dampening if hasattr(cfg, "dampening") else 0.0,
                     nesterov=cfg.nesterov if hasattr(cfg, "nesterov") else False,
                     loss_scale=loss_scale)
    else:
        raise ValueError("Unsupported optimizer.")
    return opt


class EvalCallBack(Callback):
    """
    eval callback
    """
    def __init__(self, eval_network, ds_val, eval_epoch_frq, epoch_size, metrics, result_evaluation=None):
        self.eval_network = eval_network
        self.ds_val = ds_val
        self.eval_epoch_frq = eval_epoch_frq
        self.epoch_size = epoch_size
        self.result_evaluation = result_evaluation
        self.metrics = metrics
        self.best_result = None
        self.eval_network.set_train(False)

    def epoch_end(self, run_context):
        """
        do eval in epoch end
        """
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % self.eval_epoch_frq == 0 or cur_epoch == self.epoch_size:
            result = do_eval(self.eval_network, self.ds_val, self.metrics, cur_epoch=cur_epoch)
            if self.best_result is None or self.best_result["psnr"] < result["psnr"]:
                self.best_result = result
            if get_rank_id() == 0:
                print(f"best evaluation result = {self.best_result}", flush=True)
            if isinstance(self.result_evaluation, dict):
                for k, v in result.items():
                    r_list = self.result_evaluation.get(k)
                    if r_list is None:
                        r_list = []
                        self.result_evaluation[k] = r_list
                    r_list.append(v)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    """
    run train
    """
    print(config)
    cfg = config

    init_env(cfg)

    ds_train = init_dataset(cfg, "train")
    ds_val = init_dataset(cfg, "valid")

    net = init_net(cfg)
    cfg.steps_per_epoch = ds_train.get_dataset_size()
    opt = init_opt(cfg, net)

    loss = nn.L1Loss(reduction='mean')

    eval_net = net

    model = Model(net, loss_fn=loss, optimizer=opt, amp_level=cfg.amp_level)

    metrics = {
        "psnr": PSNR(rgb_range=cfg.rgb_range, shave=True),
    }
    eval_cb = EvalCallBack(eval_net, ds_val, cfg.eval_epoch_frq, cfg.epoch_size, metrics=metrics)

    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.steps_per_epoch * cfg.save_epoch_frq,
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)
    time_cb = TimeMonitor()
    ckpoint_cb = ModelCheckpoint(prefix=f"EDSR_x{cfg.scale}_" + cfg.dataset_name, directory=cfg.ckpt_save_dir,
                                 config=config_ck)
    loss_cb = LossMonitor()
    cbs = [time_cb, ckpoint_cb, loss_cb, eval_cb]
    if get_device_num() > 1 and get_rank_id() != 0:
        cbs = [time_cb, loss_cb, eval_cb]

    model.train(cfg.epoch_size, ds_train, dataset_sink_mode=cfg.dataset_sink_mode, callbacks=cbs)
    print("train success", flush=True)

if __name__ == '__main__':
    run_train()
