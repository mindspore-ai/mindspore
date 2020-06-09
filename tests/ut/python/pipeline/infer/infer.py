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
""" infer """
from argparse import ArgumentParser
import numpy as np

from mindspore import Tensor
from ....dataset_mock import MindData

__factory = {
    "resnet50": resnet50(),
}


def parse_args():
    """ parse_args """
    parser = ArgumentParser(description="resnet50 example")

    parser.add_argument("--model", type=str, default="resnet50",
                        help="the network architecture for training or testing")
    parser.add_argument("--phase", type=str, default="test",
                        help="the phase of the model, default is test.")
    parser.add_argument("--file_path", type=str, default="/data/file/test1.txt",
                        help="data directory of training or testing")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="batch size for training or testing ")

    return parser.parse_args()


def get_model(name):
    """ get_model """
    if name not in __factory:
        raise KeyError("unknown model:", name)
    return __factory[name]


def get_dataset(batch_size=32):
    """ get_dataset """
    dataset_types = np.float32
    dataset_shapes = (batch_size, 3, 224, 224)

    dataset = MindData(size=2, batch_size=batch_size,
                       np_types=dataset_types,
                       output_shapes=dataset_shapes,
                       input_indexs=(0, 1))
    return dataset


# pylint: disable=unused-argument
def test(name, file_path, batch_size):
    """ test """
    network = get_model(name)

    batch = get_dataset(batch_size=batch_size)

    data_list = []
    for data in batch:
        data_list.append(data.asnumpy())
    batch_data = np.concatenate(data_list, axis=0).transpose((0, 3, 1, 2))
    input_tensor = Tensor(batch_data)
    print(input_tensor.shape)
    network(input_tensor)


if __name__ == '__main__':
    args = parse_args()
    if args.phase == "train":
        raise NotImplementedError
    test(args.model, args.file_path, args.batch_size)
