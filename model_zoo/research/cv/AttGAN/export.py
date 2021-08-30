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
"""export file."""
import argparse
import json
from os.path import join
import numpy as np

from mindspore import context, Tensor
from mindspore.train.serialization import export, load_param_into_net

from src.utils import resume_generator
from src.attgan import Gen

parser = argparse.ArgumentParser(description='Attribute Edit')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument('--gen_ckpt_name', type=str, default='')
parser.add_argument('--file_format', type=str, choices=["AIR", "MINDIR"], default='AIR', help='file format')
parser.add_argument('--experiment_name', dest='experiment_name', required=True)

args_ = parser.parse_args()
print(args_)

with open(join('output', args_.experiment_name, 'setting.txt'), 'r') as f:
    args = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
args.device_id = args_.device_id
args.batch_size = args_.batch_size
args.gen_ckpt_name = args_.gen_ckpt_name
args.file_format = args_.file_format

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args.device_id)

if __name__ == '__main__':

    gen = Gen(mode="test")

    para_gen = resume_generator(args, gen, args.gen_ckpt_name)
    load_param_into_net(gen, para_gen)

    input_array = Tensor(np.random.uniform(-1.0, 1.0, size=(1, 3, 128, 128)).astype(np.float32))
    input_label = Tensor(np.random.uniform(-1.0, 1.0, size=(1, 13)).astype(np.float32))

    Gen_file = f"attgan_mindir"
    export(gen, *(input_array, input_label), file_name=Gen_file, file_format=args.file_format)
