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
"""unet 310 infer preprocess dataset"""
import argparse
from src.data_loader import create_dataset
from src.config import cfg_unet


def preprocess_dataset(data_dir, result_path, cross_valid_ind=1, cfg=None):

    _, valid_dataset = create_dataset(data_dir, 1, 1, False, cross_valid_ind, False, do_crop=cfg['crop'],
                                      img_size=cfg['img_size'])

    for i, data in enumerate(valid_dataset):
        file_name = "ISBI_test_bs_1_" + str(i) + ".bin"
        file_path = result_path + file_name
        data[0].asnumpy().tofile(file_path)


def get_args():
    parser = argparse.ArgumentParser(description='Preprocess the UNet dataset ',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_url', dest='data_url', type=str, default='data/',
                        help='data directory')
    parser.add_argument('-p', '--result_path', dest='result_path', type=str, default='./preprocess_Result/',
                        help='result path')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    preprocess_dataset(data_dir=args.data_url, cross_valid_ind=cfg_unet['cross_valid_ind'], cfg=cfg_unet, result_path=
                       args.result_path)
