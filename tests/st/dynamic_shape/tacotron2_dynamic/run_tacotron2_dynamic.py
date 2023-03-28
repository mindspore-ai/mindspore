# Copyright 2023 Huawei Technologies Co., Ltd
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
'''training model'''
import math

import numpy as np
import mindspore
import mindspore.dataset as ds
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore import context
from mindspore import Model
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.nn.optim import Adam
from mindspore.common.initializer import One
from mindspore.train.callback import Callback
from mindspore.common.api import jit
from mindspore.ops import functional as F

from src.hparams import hparams as hps
from src.callback import get_lr
from src.tacotron2 import Tacotron2, Tacotron2Loss, NetWithLossClass
from src.tacotron2 import TrainStepWrap, grad_scale, clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE
from model_utils.config import config

np.random.seed(0)
mindspore.common.set_seed(1024)

context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target, max_call_depth=8000)


class TrainStepWrapNew(TrainStepWrap):
    def __init__(self, network, optimizer, scale_update_cell):
        super(TrainStepWrapNew, self).__init__(network, optimizer, scale_update_cell)
        self.enable_tuple_broaden = True

    def construct(self,
                  text_padded,
                  input_length,
                  mel_padded,
                  gate_padded,
                  text_mask,
                  mel_mask,
                  rnn_mask):
        ''' construct '''
        weights = self.weights
        loss = self.network(text_padded,
                            input_length,
                            mel_padded,
                            gate_padded,
                            text_mask,
                            mel_mask,
                            rnn_mask)
        scale_sense = self.loss_scale
        init = self.alloc_status()
        init = F.depend(init, loss)
        clear_status = self.clear_before_grad(init)
        scale_sense = F.depend(scale_sense, clear_status)
        if self.fp16_flag:
            grads = self.grad(self.network, weights)(text_padded,
                                                     input_length,
                                                     mel_padded,
                                                     gate_padded,
                                                     text_mask,
                                                     mel_mask,
                                                     rnn_mask,
                                                     self.cast(scale_sense, mindspore.float16))
        else:
            grads = self.grad(self.network, weights)(text_padded,
                                                     input_length,
                                                     mel_padded,
                                                     gate_padded,
                                                     text_mask,
                                                     mel_mask,
                                                     rnn_mask,
                                                     self.cast(scale_sense, mindspore.float32))
        grads = self.grad_reducer(grads)
        grads = self.process_grads(grads)
        cond = self.get_overflow_status(init, grads)
        overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if not overflow:
            self.optimizer(grads)
        return loss, scale_sense.value(), overflow

    @jit
    def get_overflow_status(self, init, grads):
        init = F.depend(init, grads)
        get_status = self.get_status(init)
        init = F.depend(init, get_status)
        flag_sum = self.reduce_sum(init, (0,))

        if self.is_distributed:
            flag_reduce = self.all_reduce(flag_sum)
            cond = self.less_equal(self.base, flag_reduce)
        else:
            cond = self.less_equal(self.base, flag_sum)
        return cond

    @jit
    def process_grads(self, grads):
        grads = self.hyper_map(F.partial(grad_scale, self.degree * self.loss_scale), grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        return grads


def collate(text, mel, _):
    ''' pad text sequence and mel spectrogram
    text: [text_len] * batch
    mel:  [n_mels, n_frame] * batch
    '''
    # sort
    bs = len(text)
    text_lens = [t.shape[0] for t in text]
    ids_descend = np.argsort(np.array(text_lens))[::-1]
    input_lengths = np.zeros((bs,), np.int32)
    ids = range(bs)
    for i, i_d in zip(ids, ids_descend):
        input_lengths[i] = text[i_d].shape[0]

    # pad text
    max_text_len = input_lengths[0]
    text_padded = np.zeros((bs, max_text_len), np.int32)  # vs ones
    text_mask = np.zeros((bs, max_text_len)).astype(np.bool)
    for i, i_d in zip(ids, ids_descend):
        text_len = input_lengths[i]
        text_padded[i, :text_len] = text[i_d]
        text_mask[i, :text_len] = True

    # pad mel
    n_mels = mel[0].shape[0]
    max_target_len = max([x.shape[1] for x in mel])
    if max_target_len % hps.n_frames_per_step != 0:
        max_target_len += hps.n_frames_per_step - max_target_len % hps.n_frames_per_step
    mel_padded = np.zeros((bs, n_mels, max_target_len), np.float32)
    gate_padded = np.zeros((bs, max_target_len), np.float32)
    mel_mask = np.zeros((bs, n_mels, max_target_len)).astype(np.bool)
    for i, i_d in zip(ids, ids_descend):
        cur_mel = mel[i_d]
        n_frames = cur_mel.shape[1]
        mel_padded[i, :, :n_frames] = cur_mel
        gate_padded[i, n_frames - 1:] = 1
        mel_mask[i, :, :n_frames] = True

    # rnn mask
    rnn_mask = np.zeros((bs, max_text_len, hps.encoder_embedding_dim)).astype(np.bool)
    for i in ids:
        text_len = input_lengths[i]
        rnn_mask[i, :text_len, :] = True
    ret = (text_padded, input_lengths, mel_padded, gate_padded, text_mask, mel_mask, rnn_mask)
    return ret


class LjdatasetDynamic:
    def __init__(self):
        self.dataset = []
        num_mel = hps.num_mels
        mel_lens = [80, 160, 240]
        text_lens = [16, 32, 48]
        for mel_len, text_len in zip(mel_lens, text_lens):
            mel_len_batch = np.random.randint(32, mel_len, (hps.batch_size,))
            text_len_batch = np.random.randint(6, text_len, (hps.batch_size,))
            for j in range(hps.batch_size):
                mel = np.random.randn(num_mel, mel_len_batch[j]).astype(np.float32)
                text = np.random.randint(0, 60, (text_len_batch[j],), dtype=np.int64)
                text[-1] = 148
                self.dataset.append({
                    'mel': mel,
                    'text': text,
                })

    def __getitem__(self, index):
        meta = self.dataset[index]
        text = meta['text']
        mel = meta['mel']
        return text, mel

    def __len__(self):
        return len(self.dataset)


def prepare_dataloaders_dynamic():
    '''prepare dataloaders dynamic'''
    dataset = LjdatasetDynamic()
    ds_dataset = ds.GeneratorDataset(dataset,
                                     ['text',
                                      'mel'],
                                     num_parallel_workers=4,
                                     shuffle=False
                                     )
    ds_dataset = ds_dataset.batch(hps.batch_size,
                                  per_batch_map=collate,
                                  input_columns=["text", "mel"],
                                  output_columns=['text_padded',
                                                  'input_lengths',
                                                  'mel_padded',
                                                  'gate_padded',
                                                  'text_mask',
                                                  'mel_mask',
                                                  'rnn_mask'],
                                  num_parallel_workers=4)
    return ds_dataset


class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss in NAN or INF terminating training.
    Note:
        if per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, dataset_size=-1):
        super(LossCallBack, self).__init__()
        self._dataset_size = dataset_size

    def step_end(self, run_context):
        """
        Print loss after each step
        """
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs[0].asnumpy()
        overflow = cb_params.net_outputs[2]
        scale = cb_params.net_outputs[1].asnumpy()
        percent, epoch_num = math.modf(cb_params.cur_step_num / self._dataset_size)
        if percent == 0:
            percent = 1
            epoch_num -= 1
        if not overflow:
            print("epoch: {}, current epoch percent: {}, step: {}, loss is {}, Scale: {}"
                  .format(int(epoch_num), "%.3f" % percent, cb_params.cur_step_num, loss, scale),
                  flush=True)
        else:
            print("epoch: {}, current epoch percent: {}, step: {}, loss is {}, Overflow: {}, Scale: {}"
                  .format(int(epoch_num), "%.3f" % percent, cb_params.cur_step_num, loss, overflow, scale),
                  flush=True)


def _build_training_pipeline(pre_dataset):
    ''' training '''

    epoch_num = config.epoch_num
    steps_per_epoch = pre_dataset.get_dataset_size()
    learning_rate = get_lr(config.lr, epoch_num, steps_per_epoch, steps_per_epoch * config.warmup_epochs)
    learning_rate = Tensor(learning_rate)
    scale_update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2 ** 12,
                                                   scale_factor=2,
                                                   scale_window=1000)
    net = Tacotron2()
    loss_fn = Tacotron2Loss()
    loss_net = NetWithLossClass(net, loss_fn)
    optimizer = Adam(params=net.trainable_params(), learning_rate=learning_rate)
    if hps.fp16_flag:
        loss_net.to_float(mstype.float16)
    train_net = TrainStepWrapNew(loss_net, optimizer, scale_update_cell)
    train_net.set_train()
    # inputs:
    # 'text_padded',
    # 'input_lengths',
    # 'mel_padded',
    # 'gate_padded',
    # 'text_mask',
    # 'mel_mask',
    # 'rnn_mask'
    train_net.set_inputs(Tensor(shape=[hps.batch_size, None], dtype=mstype.int32),
                         Tensor(shape=[hps.batch_size], dtype=mstype.int32, init=One()),
                         Tensor(shape=[hps.batch_size, hps.num_mels, None], dtype=mstype.float32),
                         Tensor(shape=[hps.batch_size, None], dtype=mstype.float32),
                         Tensor(shape=[hps.batch_size, None], dtype=mstype.bool_),
                         Tensor(shape=[hps.batch_size, hps.num_mels, None], dtype=mstype.bool_),
                         Tensor(shape=[hps.batch_size, None, hps.encoder_embedding_dim], dtype=mstype.bool_))
    model = Model(train_net)
    callbacks = [LossCallBack(steps_per_epoch)]
    model.train(
        epoch_num,
        pre_dataset,
        callbacks=callbacks,
        dataset_sink_mode=True)


def train_single():
    """
    Train model on single device
    """
    hps.batch_size = config.batch_size
    preprocessed_data = prepare_dataloaders_dynamic()
    _build_training_pipeline(preprocessed_data)


if __name__ == '__main__':
    train_single()
