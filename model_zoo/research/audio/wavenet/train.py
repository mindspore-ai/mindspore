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
"""train_criteo."""
import os
from os.path import join
import json
import argparse
from warnings import warn
from hparams import hparams, hparams_debug_string

from mindspore import context, Tensor
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn.optim import Adam
from mindspore.nn import TrainOneStepCell
from mindspore.train import Model
from src.lr_generator import get_lr
from src.dataset import get_data_loaders
from src.loss import NetWithLossClass
from src.callback import Monitor
from wavenet_vocoder import WaveNet
from wavenet_vocoder.util import is_mulaw_quantize, is_scalar_input

parser = argparse.ArgumentParser(description='TTS training')
parser.add_argument('--data_path', type=str, required=True, default='',
                    help='Directory contains preprocessed features.')
parser.add_argument('--preset', type=str, required=True, default='', help='Path of preset parameters (json).')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_test',
                    help='Directory where to save model checkpoints [default: checkpoints].')
parser.add_argument('--checkpoint', type=str, default='', help='Restore model from checkpoint path if given.')
parser.add_argument('--speaker_id', type=str, default='',
                    help=' Use specific speaker of data in case for multi-speaker datasets.')
parser.add_argument('--platform', type=str, default='GPU', choices=('GPU', 'CPU'),
                    help='run platform, support GPU and CPU. Default: GPU')
parser.add_argument('--is_distributed', action="store_true", default=False, help='Distributed training')
args = parser.parse_args()

if __name__ == '__main__':
    if args.is_distributed:
        init('nccl')
        rank_id = get_rank()
        group_size = get_group_size()
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False)
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=get_group_size(), parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=args.platform, save_graphs=False)
        rank_id = 0
        group_size = 1

    speaker_id = int(args.speaker_id) if args.speaker_id != '' else None
    if args.preset is not None:
        with open(args.preset) as f:
            hparams.parse_json(f.read())

    assert hparams.name == "wavenet_vocoder"
    print(hparams_debug_string())
    fs = hparams.sample_rate
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    output_json_path = join(args.checkpoint_dir, "hparams.json")
    with open(output_json_path, "w") as f:
        json.dump(hparams.values(), f, indent=2)

    data_loaders = get_data_loaders(args.data_path, args.speaker_id, hparams=hparams, rank_id=rank_id,
                                    group_size=group_size)
    step_size_per_epoch = data_loaders.get_dataset_size()

    if is_mulaw_quantize(hparams.input_type):
        if hparams.out_channels != hparams.quantize_channels:
            raise RuntimeError(
                "out_channels must equal to quantize_chennels if input_type is 'mulaw-quantize'")
    if hparams.upsample_conditional_features and hparams.cin_channels < 0:
        s = "Upsample conv layers were specified while local conditioning disabled. "
        s += "Notice that upsample conv layers will never be used."
        warn(s)

    upsample_params = hparams.upsample_params
    upsample_params["cin_channels"] = hparams.cin_channels
    upsample_params["cin_pad"] = hparams.cin_pad
    model = WaveNet(
        out_channels=hparams.out_channels,
        layers=hparams.layers,
        stacks=hparams.stacks,
        residual_channels=hparams.residual_channels,
        gate_channels=hparams.gate_channels,
        skip_out_channels=hparams.skip_out_channels,
        cin_channels=hparams.cin_channels,
        gin_channels=hparams.gin_channels,
        n_speakers=hparams.n_speakers,
        dropout=hparams.dropout,
        kernel_size=hparams.kernel_size,
        cin_pad=hparams.cin_pad,
        upsample_conditional_features=hparams.upsample_conditional_features,
        upsample_params=upsample_params,
        scalar_input=is_scalar_input(hparams.input_type),
        output_distribution=hparams.output_distribution,
    )
    loss_net = NetWithLossClass(model, hparams)
    lr = get_lr(hparams.optimizer_params["lr"], hparams.nepochs, step_size_per_epoch)
    lr = Tensor(lr)

    if args.checkpoint != '':
        param_dict = load_checkpoint(args.pre_trained_model_path)
        load_param_into_net(model, param_dict)
        print('Successfully loading the pre-trained model')

    weights = model.trainable_params()
    optimizer = Adam(weights, learning_rate=lr, loss_scale=1024.)
    train_net = TrainOneStepCell(loss_net, optimizer)

    model = Model(train_net)
    lr_cb = Monitor(lr)
    callback_list = [lr_cb]
    if args.is_distributed:
        ckpt_path = os.path.join(args.checkpoint_dir, 'ckpt_' + str(get_rank()) + '/')
    else:
        ckpt_path = args.checkpoint_dir
    config_ck = CheckpointConfig(save_checkpoint_steps=step_size_per_epoch, keep_checkpoint_max=10)
    ckpt_cb = ModelCheckpoint(prefix='wavenet', directory=ckpt_path, config=config_ck)
    callback_list.append(ckpt_cb)
    model.train(hparams.nepochs, data_loaders, callbacks=callback_list, dataset_sink_mode=False)
