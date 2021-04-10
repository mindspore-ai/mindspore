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
"""export mindir."""
import json
from os.path import join
import argparse
from warnings import warn
import numpy as np
from hparams import hparams, hparams_debug_string
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export
from wavenet_vocoder import WaveNet
from wavenet_vocoder.util import is_mulaw_quantize, is_scalar_input
from src.loss import PredictNet

parser = argparse.ArgumentParser(description='TTS training')
parser.add_argument('--preset', type=str, default='', help='Path of preset parameters (json).')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_test',
                    help='Directory where to save model checkpoints [default: checkpoints].')
parser.add_argument('--speaker_id', type=str, default='',
                    help=' Use specific speaker of data in case for multi-speaker datasets.')
parser.add_argument('--pretrain_ckpt', type=str, default='', help='Pretrained checkpoint path')
parser.add_argument('--platform', type=str, default='GPU', choices=('GPU', 'CPU'),
                    help='run platform, support GPU and CPU. Default: GPU')
args = parser.parse_args()

if __name__ == '__main__':

    context.set_context(mode=context.GRAPH_MODE, device_target=args.platform, save_graphs=False)

    speaker_id = int(args.speaker_id) if args.speaker_id != '' else None
    if args.preset is not None:
        with open(args.preset) as f:
            hparams.parse_json(f.read())

    assert hparams.name == "wavenet_vocoder"
    print(hparams_debug_string())

    fs = hparams.sample_rate
    output_json_path = join(args.checkpoint_dir, "hparams.json")
    with open(output_json_path, "w") as f:
        json.dump(hparams.values(), f, indent=2)

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

    Net = PredictNet(model)
    Net.set_train(False)
    param_dict = load_checkpoint(args.pretrain_ckpt)
    load_param_into_net(model, param_dict)
    print('Successfully loading the pre-trained model')

    if is_mulaw_quantize(hparams.input_type):
        x = np.array(np.random.random((2, 256, 10240)), dtype=np.float32)
    else:
        x = np.array(np.random.random((2, 1, 10240)), dtype=np.float32)
    c = np.array(np.random.random((2, 80, 44)), dtype=np.float32)
    g = np.array([0, 0], dtype=np.int64)

    export(Net, Tensor(x), Tensor(c), Tensor(g), file_name="WaveNet", file_format='MINDIR')
