# coding=utf-8
#
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

import argparse
import base64
import json
import os

import numpy as np

from multiclass_loader import MultiClassLoader
from sdk_infer_wrapper import SDKInferWrapper


def _parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="path of dataset directory")
    parser.add_argument("--pipeline", type=str, required=True,
                        help="path of pipeline file")
    parser.add_argument("--output_dir", type=str, default="./infer_result",
                        help="path of output directory")
    return parser.parse_args()


def _parse_output_data(output_data):
    infer_result_data = json.loads(output_data.data.decode())
    content = json.loads(infer_result_data['metaData'][0]['content'])
    tensor_vec = content['tensorPackageVec'][0]['tensorVec'][0]
    data_str = tensor_vec['dataStr']
    tensor_shape = tensor_vec['tensorShape']
    infer_array = np.frombuffer(base64.b64decode(data_str), dtype=np.float32)
    return infer_array.reshape(tensor_shape)


def main():
    args = _parser_args()
    sdk_infer = SDKInferWrapper()
    sdk_infer.load_pipeline(args.pipeline)
    data_loader = MultiClassLoader(args.dataset_dir)

    for image_id, image, _ in data_loader.iter_dataset():
        output_data = sdk_infer.do_infer(image)
        output_tensor = _parse_output_data(output_data)
        os.makedirs(args.output_dir, exist_ok=True)
        np.save(os.path.join(args.output_dir, f"{image_id}"), output_tensor[0])


if __name__ == "__main__":
    main()
