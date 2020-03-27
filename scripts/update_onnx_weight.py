#!/usr/bin/env python3
# coding=UTF-8
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
# ============================================================================
"""
Function:
    Use checkpoint file and onnx file as inputs, create a new onnx with Initializer's value from checkpoint file
Usage:
    python update_onnx_weight.py onnx_file checkpoint_file [output_file]
"""
import sys
from onnx import onnx_pb
from mindspore.train.serialization import load_checkpoint


def update_onnx_initializer(onnx_file, ckpt_file, output_file):
    "Update onnx initializer."
    with open(onnx_file, 'rb') as f:
        data = f.read()
    model = onnx_pb.ModelProto()
    model.ParseFromString(data)
    initializer = model.graph.initializer
    param_dict = load_checkpoint(ckpt_file)

    for i, _ in enumerate(initializer):
        item = initializer[i]
        #print(item.name, item.data_type, item.dims, len(item.raw_data))
        if not item.name in param_dict:
            print(f"Warning: Can not find '{item.name}' in checkpoint parameters dictionary")
            continue
        weight = param_dict[item.name].data.asnumpy()
        bin_data = weight.tobytes()
        if len(item.raw_data) != len(bin_data):
            print(f"Warning: Size of weight from checkpoint is different from original size, ignore it")
            continue
        item.raw_data = bin_data

    pb_msg = model.SerializeToString()
    with open(output_file, 'wb') as f:
        f.write(pb_msg)

    print(f'Graph name: {model.graph.name}')
    print(f'Initializer length: {len(initializer)}')
    print(f'Checkpoint dict length: {len(param_dict)}')
    print(f'The new weights have been written to file {output_file} successfully')


def main():
    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} onnx_file checkpoint_file [output_file]')
        sys.exit(1)
    onnx_file = sys.argv[1]
    ckpt_file = sys.argv[2]
    output_file = f'new_{onnx_file}' if len(sys.argv) == 3 else sys.argv[3]
    update_onnx_initializer(onnx_file, ckpt_file, output_file)


if __name__ == '__main__':
    main()
