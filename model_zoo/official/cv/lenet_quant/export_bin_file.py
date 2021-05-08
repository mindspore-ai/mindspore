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
export mnist dataset to bin.
"""
import os
import argparse
from mindspore import context
from src.dataset import create_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='MNIST to bin')
    parser.add_argument('--device_target', type=str, default="Ascend",
                        choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--dataset_dir', type=str, default='', help='dataset path')
    parser.add_argument('--save_dir', type=str, default='', help='path to save bin file')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for bin')
    args_, _ = parser.parse_known_args()
    return args_

if __name__ == "__main__":
    args = parse_args()
    os.environ["RANK_SIZE"] = '1'
    os.environ["RANK_ID"] = '0'
    device_id = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') else 0
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=device_id)
    mnist_path = os.path.join(args.dataset_dir, 'test')
    batch_size = args.batch_size
    save_dir = os.path.join(args.save_dir, 'lenet_quant_mnist_310_infer_data')
    folder = os.path.join(save_dir, 'mnist_bs_' + str(batch_size) + '_bin')
    if not os.path.exists(folder):
        os.makedirs(folder)
    ds = create_dataset(mnist_path, batch_size)
    iter_num = 0
    label_file = os.path.join(save_dir, './mnist_bs_' + str(batch_size) + '_label.txt')
    with open(label_file, 'w') as f:
        for data in ds.create_dict_iterator():
            image = data['image']
            label = data['label']
            file_name = "mnist_" + str(iter_num) + ".bin"
            file_path = folder + "/" + file_name
            image.asnumpy().tofile(file_path)
            f.write(file_name)
            for i in label:
                f.write(',' + str(i))
            f.write('\n')
            iter_num += 1
    print("=====iter_num:{}=====".format(iter_num))
    print("=====image_data:{}=====".format(image))
    print("=====label_data:{}=====".format(label))
