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
# ===========================================================================
"""Callback."""
import time

from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import TimeMonitor
from mindspore.train.callback import CheckpointConfig, Callback


class ProgressMonitor(Callback):
    '''Progress Monitor.'''
    def __init__(self, args):
        super(ProgressMonitor, self).__init__()
        self.args = args
        self.epoch_start_time = 0
        self.step_start_time = 0
        self.globe_step_cnt = 0
        self.local_step_cnt = 0
        self.ckpt_history = []

    def begin(self, run_context):
        if not self.args.epoch_cnt:
            self.args.logger.info('start network train...')
        if run_context is None:
            pass

    def step_begin(self, run_context):
        if self.local_step_cnt == 0:
            self.step_start_time = time.time()
        if run_context is None:
            pass

    def step_end(self, run_context):
        '''Callback when step end.'''
        if self.local_step_cnt % self.args.log_interval == 0 and self.local_step_cnt > 0:
            cb_params = run_context.original_args()
            time_used = time.time() - self.step_start_time
            fps_mean = self.args.per_batch_size * self.args.log_interval / time_used
            self.args.logger.info('epoch[{}], iter[{}], loss:{}, mean_wps:{:.2f} wavs/sec'.format(self.args.epoch_cnt,
                                                                                                  self.globe_step_cnt +
                                                                                                  self.local_step_cnt,
                                                                                                  cb_params.net_outputs,
                                                                                                  fps_mean))
            self.step_start_time = time.time()
        self.local_step_cnt += 1

    def epoch_begin(self, run_context):
        self.epoch_start_time = time.time()
        if run_context is None:
            pass

    def epoch_end(self, run_context):
        '''Callback when epoch end.'''
        cb_params = run_context.original_args()
        self.globe_step_cnt = self.args.steps_per_epoch * (self.args.epoch_cnt + 1) - 1

        time_used = time.time() - self.epoch_start_time
        fps_mean = self.args.per_batch_size * self.args.steps_per_epoch / time_used
        self.args.logger.info(
            'epoch[{}], iter[{}], loss:{}, mean_wps:{:.2f} wavs/sec'.format(self.args.epoch_cnt, self.globe_step_cnt,
                                                                            cb_params.net_outputs, fps_mean))
        self.args.epoch_cnt += 1
        self.local_step_cnt = 0

    def end(self, run_context):
        pass


def callback_func(args, cb, prefix):
    callbacks = [cb]
    if args.rank_save_ckpt_flag:
        ckpt_max_num = args.max_epoch * args.steps_per_epoch // args.ckpt_interval
        ckpt_config = CheckpointConfig(save_checkpoint_steps=args.ckpt_interval, keep_checkpoint_max=ckpt_max_num)
        ckpt_cb = ModelCheckpoint(config=ckpt_config, directory=args.outputs_dir, prefix=prefix)
        callbacks.append(ckpt_cb)
    callbacks.append(TimeMonitor(args.per_batch_size))
    return callbacks
