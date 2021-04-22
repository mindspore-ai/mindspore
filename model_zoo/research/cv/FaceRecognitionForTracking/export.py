# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""Convert ckpt to air/mindir."""
import os
import argparse
import numpy as np

from mindspore import context
from mindspore import Tensor
from mindspore.train.serialization import export, load_checkpoint, load_param_into_net

from src.reid import SphereNet_float32


def main(args):
    network = SphereNet_float32(num_layers=12, feature_dim=128, shape=(96, 64))
    ckpt_path = args.pretrained
    if os.path.isfile(ckpt_path):
        param_dict = load_checkpoint(ckpt_path)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('model.'):
                param_dict_new[key[6:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(network, param_dict_new)
        print('-----------------------load model success-----------------------')
    else:
        print('-----------------------load model failed -----------------------')

    if args.device_target == 'CPU':
        network.add_flags_recursive(fp32=True)
    else:
        network.add_flags_recursive(fp16=True)
    network.set_train(False)

    input_data = np.random.uniform(low=0, high=1.0, size=(args.batch_size, 3, 96, 64)).astype(np.float32)
    tensor_input_data = Tensor(input_data)

    export(network, tensor_input_data, file_name=args.file_name, file_format=args.file_format)
    print('-----------------------export model success-----------------------')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Convert ckpt to air/mindir')
    parser.add_argument('--pretrained', type=str, default='', help='pretrained model to load')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--device_target', type=str, choices=['Ascend', 'GPU', 'CPU'], default='Ascend',
                        help='device_target')
    parser.add_argument('--file_name', type=str, default='FaceRecognitionForTracking', help='output file name')
    parser.add_argument('--file_format', type=str, choices=['AIR', 'ONNX', 'MINDIR'], default='AIR', help='file format')

    arg = parser.parse_args()

    if arg.device_target == 'Ascend':
        devid = int(os.getenv('DEVICE_ID'))
        context.set_context(device_id=devid)

    context.set_context(mode=context.GRAPH_MODE, device_target=arg.device_target)

    main(arg)
